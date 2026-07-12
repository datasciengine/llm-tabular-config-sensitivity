"""Few-shot in-context generator. No training -> cheap.

Builds a prompt from `n_shots` real rows serialized per `config`, asks a small
open instruct model for new rows, parses them back to the schema. Highly
config-sensitive (prompt-driven), which strengthens the study.

the project spec sec 12.7: generate in chunks, parse -> validate -> retry, drop invalid
rows, up to max_oversample x n attempts. The ICL valid-row YIELD is logged and is
itself a finding.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pandas as pd

from .base import BaseGenerator, set_all_seeds
from .. import configs


def _hf_id(name: str) -> str:
    return name if "/" in name else f"Qwen/{name}"


# ----------------------------------------------------------------------------
# Per-scheme line parser (reverse of configs.serialize_row)
# ----------------------------------------------------------------------------
def parse_line(line: str, scheme: str, cols_in_order: list[str]) -> dict | None:
    # Strip ONLY a leading list/enumeration marker ("1. ", "2) ", "- ", "* ").
    # A blanket strip of digits/periods from both ends would eat the last column's
    # value in the sentence scheme ("... The Outcome is 0." -> "... The Outcome is")
    # and the first value in the compact scheme. Match the marker pattern instead.
    line = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", "", line.strip())
    if not line:
        return None
    out: dict[str, str] = {}
    if scheme == "sentence":
        for clause in re.split(r"\.\s+|\.$", line):
            m = re.match(r"\s*[Tt]he (.+?) is (.+?)\.?\s*$", clause.strip())
            if m and m.group(1) in cols_in_order:
                out[m.group(1)] = m.group(2).strip()
    elif scheme == "keyvalue":
        for part in line.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() in cols_in_order:
                    out[k.strip()] = v.strip()
    elif scheme == "compact":
        fields = line.split("|")
        if len(fields) == len(cols_in_order):
            out = {c: fields[i].strip() for i, c in enumerate(cols_in_order)}
    else:
        raise ValueError(scheme)
    return out if set(out) == set(cols_in_order) else None


class ICLGenerator(BaseGenerator):
    config_sensitive = True

    def fit(self, train_df, seed, config, ds_cfg):
        set_all_seeds(seed)
        self.train_df = train_df.reset_index(drop=True)
        self.config = config
        self.ds_cfg = ds_cfg
        self.columns = list(train_df.columns)               # canonical order
        ordered = configs.apply_column_order(train_df, config["column_order"])
        self.cols_in_order = list(ordered.columns)
        ct = self._types(train_df, ds_cfg.get("target"))
        self.numeric_cols, self.categorical_cols = ct
        self.last_yield = None
        # No training. The model is loaded lazily at sample time.
        self._model = None
        self._tok = None

    @staticmethod
    def _types(df, target):
        # A column is numeric iff it has a genuine numeric dtype. Use is_numeric_dtype
        # (NOT `dtype == object`) so it is robust to pandas reading string columns as the
        # newer 'string' dtype instead of 'object' — otherwise categorical columns get
        # mis-typed as numeric and _coerce float()-rejects every row (0 yield on
        # mixed-type tables like adult/student; all-numeric diabetes was unaffected).
        numeric, categorical = [], []
        for c in df.columns:
            if c == target or not pd.api.types.is_numeric_dtype(df[c]):
                categorical.append(c)
            else:
                numeric.append(c)
        return numeric, categorical

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = _hf_id(self.base_model)
        # Device: CUDA on the cloud GPU; otherwise CPU. We deliberately avoid Apple
        # MPS here — generation tripped a native MPS assertion ("NDArray > 2**32")
        # on Qwen. Override with ICL_DEVICE if needed. (ICL is cheap; CPU is fine.)
        self._device = os.environ.get("ICL_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._tok = AutoTokenizer.from_pretrained(hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype="auto").to(self._device)
        self._model.eval()

    def _build_prompt(self, n_request: int) -> str:
        n_shots = int(self.params.get("n_shots", 20))
        shots = self.train_df.sample(n=min(n_shots, len(self.train_df)),
                                     random_state=0)
        header, rows = configs.serialize_frame(shots, self.config)
        scheme = self.config["serialization"]
        fmt_hint = {
            "sentence": "Each record is one line like: The <column> is <value>. (one sentence per column)",
            "keyvalue": "Each record is one line like: col1=value1, col2=value2, ...",
            "compact": f"Each record is one line of '|'-separated values in this exact column order: {header}",
        }[scheme]
        examples = "\n".join(rows)
        return (
            f"You generate synthetic tabular records that look like real ones.\n"
            f"Columns: {', '.join(self.cols_in_order)}.\n"
            f"Format: {fmt_hint}\n"
            f"Here are {len(rows)} example records:\n{examples}\n\n"
            f"Now output {n_request} NEW records in the EXACT same format, one per line. "
            f"Output ONLY the records — no Python code, no commentary, no numbering, "
            f"no markdown fences."
        )

    def _prefill(self) -> str:
        """A scheme-appropriate, value-agnostic opening for the assistant turn. The
        0.5B instruct model otherwise tends to answer with a Python snippet instead of
        emitting records; anchoring the first tokens forces it into the row format.
        Returns '' for compact (values-only -> no safe anchor)."""
        scheme = self.config["serialization"]
        first = self.cols_in_order[0]
        if scheme == "sentence":
            return f"The {first} is "
        if scheme == "keyvalue":
            return f"{first}="
        return ""

    def sample(self, n, seed):
        """Dispatch to the vLLM engine (fast, for the wide-table grid) when
        ICL_BACKEND=vllm; otherwise the reference HF `generate` path. Both share the
        EXACT prompt, sampling params, parser, padding and yield accounting — only the
        inference engine differs (not a scientific variable)."""
        if os.environ.get("ICL_BACKEND", "").lower() == "vllm":
            return self._sample_vllm(n, seed)
        return self._sample_hf(n, seed)

    def _sample_hf(self, n, seed):
        set_all_seeds(seed)
        self._load_model()
        import torch

        scheme = self.config["serialization"]
        chunk = int(self.params.get("chunk_size", 32))
        temperature = float(self.params.get("temperature", 0.8))
        max_oversample = float(self.params.get("max_oversample", 3))
        # gen_batch: how many independent sequences to draw per generate() call. On a
        # GPU these run in parallel, so batch>1 gives a near-linear throughput win for
        # the n=5000 grid without changing the sampling distribution (each sequence is
        # an independent sample). Defaults to 1 (the CPU/smoke-validated path).
        batch = max(1, int(os.environ.get("ICL_GEN_BATCH", self.params.get("gen_batch", 1))))
        max_attempts = max(1, math.ceil(max_oversample * n / (chunk * batch)))
        # Token budget must scale with schema WIDTH: a fixed chunk*64 gives ~64 tokens
        # per record, enough for narrow tables (diabetes, 9 cols) but it TRUNCATES wide
        # records (adult 15 / student 33 cols) mid-line -> those rows fail to parse and
        # yield collapses for reasons unrelated to model capability. Scale ~14 tok/col
        # (covers the verbose "The <col> is <val>." sentence scheme) and cap the total
        # so a single generate() call stays bounded. Env ICL_MAX_NEW_TOKENS overrides.
        n_cols = len(self.cols_in_order)
        per_record_tokens = max(48, 14 * n_cols)
        max_new = int(os.environ.get("ICL_MAX_NEW_TOKENS", min(chunk * per_record_tokens, 4096)))

        valid: list[dict] = []
        n_parsed = 0
        for attempt in range(max_attempts):
            if len(valid) >= n:
                break
            prompt = self._build_prompt(chunk)
            messages = [{"role": "user", "content": prompt}]
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prefill = self._prefill()
            text += prefill                          # anchor the assistant into row format
            inputs = self._tok(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, do_sample=True, temperature=temperature, top_p=0.95,
                    max_new_tokens=max_new, pad_token_id=self._tok.eos_token_id,
                    num_return_sequences=batch,
                )
            in_len = inputs["input_ids"].shape[1]
            for seq in out:                          # batch independent generations
                gen = prefill + self._tok.decode(seq[in_len:], skip_special_tokens=True)
                for line in gen.splitlines():
                    row = parse_line(line, scheme, self.cols_in_order)
                    if row is None:
                        continue
                    n_parsed += 1
                    coerced = self._coerce(row)
                    if coerced is not None:
                        valid.append(coerced)

        self.last_yield = (len(valid) / n_parsed) if n_parsed else 0.0
        if not valid:
            raise RuntimeError(
                f"ICL produced 0 valid rows (scheme={scheme}); yield=0. "
                "Recorded as a failed cell."
            )
        df = pd.DataFrame(valid)[self.columns]
        if len(df) < n:                       # pad with replacement from valid rows
            df = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
        else:
            df = df.iloc[:n].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------------
    # vLLM backend (throughput). Identical prompt / sampling / parser / padding /
    # yield accounting as _sample_hf; only the inference engine differs.
    # ------------------------------------------------------------------------
    def _load_vllm(self):
        if getattr(self, "_vllm", None) is not None:
            # Engine may have been INJECTED (shared across cells by a load-once driver);
            # make sure the tokenizer is available too, then reuse it (no reload).
            if getattr(self, "_tok", None) is None:
                self._tok = self._vllm.get_tokenizer()
            return
        # vLLM V1 launches an EngineCore worker; the default 'fork' clashes with a CUDA
        # context already present in this process -> "Cannot re-initialize CUDA in forked
        # subprocess". Force 'spawn' (fresh worker interpreter).
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        hf_id = _hf_id(self.base_model)
        # enforce_eager: skip CUDA-graph capture -> fast startup (engine loads once per
        # cell subprocess). max_model_len bounds KV cache for the wide few-shot prompt
        # (20 shots x wide rows) + generation.
        self._vllm = LLM(
            model=hf_id,
            dtype="bfloat16",
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.90")),
            enforce_eager=os.environ.get("VLLM_EAGER", "1") == "1",  # 0 = CUDA graphs (big GPU)
            max_model_len=int(os.environ.get("VLLM_MAX_LEN", "16384")),
        )
        self._tok = self._vllm.get_tokenizer()

    def _sample_vllm(self, n, seed):
        # Seed python/numpy only (NOT torch.cuda) so we do not create a CUDA context in
        # THIS process before vLLM starts its worker. vLLM generation reproducibility
        # comes from SamplingParams(seed=...); padding uses random_state=seed.
        import random
        random.seed(seed)
        np.random.seed(seed)
        self._load_vllm()
        from vllm import SamplingParams

        scheme = self.config["serialization"]
        chunk = int(self.params.get("chunk_size", 32))
        temperature = float(self.params.get("temperature", 0.8))
        max_oversample = float(self.params.get("max_oversample", 3))
        n_cols = len(self.cols_in_order)
        per_record_tokens = max(48, 14 * n_cols)               # same as HF path
        max_new = int(os.environ.get("ICL_MAX_NEW_TOKENS", min(chunk * per_record_tokens, 4096)))
        # How many records realistically fit one sequence's token budget; request enough
        # sequences to reach n at full yield, repeat up to max_oversample rounds (stopping
        # early) for the shortfall. Same budget as HF, but vLLM batches a whole round in
        # one call — that is the throughput win.
        rounds = max(1, math.ceil(max_oversample))

        # Prompt is deterministic (fixed shots, random_state=0) -> build once.
        prompt = self._build_prompt(chunk)
        messages = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prefill = self._prefill()
        text += prefill                                        # same assistant anchor as HF

        valid: list[dict] = []
        n_parsed = 0
        for r in range(rounds):
            if len(valid) >= n:
                break
            # Size THIS round to the remaining shortfall (~chunk records/sequence), so a
            # 2nd/3rd round only TOPS UP what's missing instead of re-generating the full
            # n. Big time saver on wide tables where round 1 nearly fills n.
            need = n - len(valid)
            seqs = max(1, math.ceil(need / chunk))
            sp = SamplingParams(
                n=seqs, temperature=temperature, top_p=0.95,
                max_tokens=max_new, seed=int(seed) * 1000 + r,  # reproducible; distinct per round
            )
            outs = self._vllm.generate([text], sp, use_tqdm=True)  # show per-round progress
            for o in outs[0].outputs:
                gen = prefill + o.text
                for line in gen.splitlines():
                    row = parse_line(line, scheme, self.cols_in_order)
                    if row is None:
                        continue
                    n_parsed += 1
                    coerced = self._coerce(row)
                    if coerced is not None:
                        valid.append(coerced)

        self.last_yield = (len(valid) / n_parsed) if n_parsed else 0.0
        if not valid:
            raise RuntimeError(
                f"ICL(vLLM) produced 0 valid rows (scheme={scheme}); yield=0. "
                "Recorded as a failed cell."
            )
        df = pd.DataFrame(valid)[self.columns]
        if len(df) < n:                       # pad with replacement (same as HF)
            df = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
        else:
            df = df.iloc[:n].reset_index(drop=True)
        return df

    def _coerce(self, row: dict) -> dict | None:
        """Validate + cast a parsed row to the schema. Numeric cols must parse as
        numbers; otherwise the row is dropped."""
        out = {}
        for c in self.columns:
            v = row.get(c)
            if c in self.numeric_cols:
                try:
                    f = float(str(v).replace(",", ""))
                except (TypeError, ValueError):
                    return None
                out[c] = int(f) if pd.api.types.is_integer_dtype(self.train_df[c]) and float(f).is_integer() else f
            else:
                out[c] = str(v)
        return out
