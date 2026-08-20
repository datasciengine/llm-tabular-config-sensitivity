"""GReaT adapter (Borisov et al.). Install: pip install be-great.

Re-serializes training data per `config` (serialization/order/format) BEFORE
fine-tuning the small base model (distilgpt2).

How each config axis reaches GReaT
----------------------------------
- numeric_format : applied to the dataframe (configs.format_numbers) before fit.
- column_order   : applied to the dataframe before fit. NOTE GReaT also permutes
                   columns internally each row (its documented augmentation), so it
                   is largely order-invariant by design — an honest finding, kept
                   per the project spec ("use documented defaults").
- serialization  : 'sentence' is realized by GReaT's NATIVE "<col> is <val>"
                   template (the format its parser AND start-sampler are built
                   around). 'keyvalue'/'compact' are realized by a self-contained
                   custom-template fine-tune (see below) that reuses the project's
                   own serializer (configs.serialize_frame) and parser
                   (icl.parse_line), so GReaT's coupled encode/decode/start-sampler
                   never has to be monkey-patched.

Serialization axis (R2-1): two fine-tune paths behind one interface
-------------------------------------------------------------------
- NATIVE path (scheme='sentence', default): the be_great GReaT class, unchanged.
  This keeps the paper's existing sentence/column-order/numeric-format cells valid.
- CUSTOM path (scheme in {'keyvalue','compact'}, or GREAT_CUSTOM_ALL=1 for a clean
  within-pipeline 3-way serialization comparison): fine-tune the SAME base model
  (distilgpt2) on rows serialized by configs.serialize_frame — exactly the templates
  ICL uses — then sample text and parse it back with icl.parse_line. This is still
  GReaT's paradigm (fine-tune an autoregressive LM on serialized rows); only the
  serialization template — the variable under study — changes. For 'compact'
  (values-only) decoding is positional, so the custom path fixes the column order and
  does NOT shuffle columns.
"""
from __future__ import annotations

import os

import pandas as pd

from .base import BaseGenerator, set_all_seeds
from .. import configs
from .icl import parse_line

_NATIVE_SCHEME = "sentence"   # GReaT's native template stands in for 'sentence'


def _numeric_categorical(df: pd.DataFrame, target: str | None):
    """Same typing rule as the ICL generator: a column is numeric iff it has a genuine
    numeric dtype and is not the target."""
    numeric, categorical = [], []
    for c in df.columns:
        if c == target or not pd.api.types.is_numeric_dtype(df[c]):
            categorical.append(c)
        else:
            numeric.append(c)
    return numeric, categorical


class GReaTGenerator(BaseGenerator):
    config_sensitive = True

    # ------------------------------------------------------------------ fit
    def fit(self, train_df, seed, config, ds_cfg):
        set_all_seeds(seed)
        self.config = config
        self.ds_cfg = ds_cfg
        self.columns = list(train_df.columns)               # canonical order
        self._dtypes = {c: train_df[c].dtype for c in train_df.columns}
        scheme = config["serialization"]
        use_custom = scheme != _NATIVE_SCHEME or os.environ.get("GREAT_CUSTOM_ALL") == "1"
        self._custom = use_custom
        if use_custom:
            return self._fit_custom(train_df, seed, config, ds_cfg)
        return self._fit_native(train_df, seed, config, ds_cfg)

    def _fit_native(self, train_df, seed, config, ds_cfg):
        """Original be_great path (serialization='sentence')."""
        df = configs.apply_column_order(train_df, config["column_order"])
        df = configs.format_numbers(df, config["numeric_format"])

        from be_great import GReaT

        epochs = int(self.params.get("epochs", 50))
        batch_size = int(self.params.get("batch_size", 32))
        exp_dir = os.environ.get("GREAT_EXP_DIR", "results/.great_trainer")
        self.model = GReaT(
            self.base_model or "distilgpt2",
            experiment_dir=exp_dir,
            epochs=epochs,
            batch_size=batch_size,
            report_to=[],
            save_strategy="no",
            logging_strategy="no",
            fp16=False,
            dataloader_num_workers=0,
            seed=seed,
        )
        self.model.fit(df)

    def _fit_custom(self, train_df, seed, config, ds_cfg):
        """Custom-template fine-tune for keyvalue/compact (and, under GREAT_CUSTOM_ALL,
        sentence). Reuses configs.serialize_frame so the template is identical to ICL's."""
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.scheme = config["serialization"]
        ordered = configs.apply_column_order(train_df, config["column_order"])
        self.cols_in_order = list(ordered.columns)
        self.numeric_cols, _ = _numeric_categorical(train_df, ds_cfg.get("target"))
        # One serialized text line per training row (column order + numeric format
        # baked in by serialize_frame).
        self._header, rows = configs.serialize_frame(train_df, config)
        self.last_yield = None
        # Generation is primed with a scheme-appropriate LINE-START so the model
        # continues from the same distribution it saw at position 0 during training
        # (GReaT's "start sampler" idea). For 'compact' (values-only, no key anchor)
        # we sample a real first-column value per attempt; sentence/keyvalue use a
        # value-agnostic column-name anchor (no copy bias).
        self._compact_starts = [r.split("|", 1)[0] for r in rows] if self.scheme == "compact" else None

        hf_id = self.base_model or "distilgpt2"
        tok = AutoTokenizer.from_pretrained(hf_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_id)
        model.resize_token_embeddings(len(tok))

        eos = tok.eos_token
        max_len = int(self.params.get("great_max_len", 128))
        lines = [(r + eos) for r in rows]
        enc = tok(lines, truncation=True, max_length=max_len)

        class _RowDS(Dataset):
            def __init__(self, encodings):
                self.ids = encodings["input_ids"]

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, i):
                ids = self.ids[i]
                return {"input_ids": ids, "attention_mask": [1] * len(ids),
                        "labels": list(ids)}

        def _collate(batch):
            maxlen = max(len(b["input_ids"]) for b in batch)
            pad_id = tok.pad_token_id
            out = {"input_ids": [], "attention_mask": [], "labels": []}
            for b in batch:
                n_pad = maxlen - len(b["input_ids"])
                out["input_ids"].append(b["input_ids"] + [pad_id] * n_pad)
                out["attention_mask"].append(b["attention_mask"] + [0] * n_pad)
                # ignore pad positions in the loss
                out["labels"].append(b["labels"] + [-100] * n_pad)
            return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}

        epochs = int(self.params.get("epochs", 50))
        batch_size = int(self.params.get("batch_size", 32))
        lr = float(self.params.get("lr", 5e-5))
        # Device: CUDA on the cloud GPU; CPU otherwise. We avoid Apple MPS for training
        # (occasional native asserts on small models); override with GREAT_DEVICE.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = os.environ.get("GREAT_DEVICE") or device
        model.to(device)

        # Plain manual training loop (no HF Trainer -> no accelerate dependency, fully
        # CPU-testable). distilgpt2 is tiny; this is the whole fine-tune.
        gen = torch.Generator().manual_seed(int(seed))
        loader = DataLoader(_RowDS(enc), batch_size=batch_size, shuffle=True,
                            collate_fn=_collate, generator=gen)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                loss = model(**b).loss
                loss.backward()
                opt.step()
                opt.zero_grad()

        self._tok = tok
        self.model = model
        self._device = device
        self.model.eval()

    # --------------------------------------------------------------- sample
    def sample(self, n, seed):
        if getattr(self, "_custom", False):
            return self._sample_custom(n, seed)
        return self._sample_native(n, seed)

    def _sample_native(self, n, seed):
        set_all_seeds(seed)
        import torch

        device = (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        df = self.model.sample(n_samples=n, device=device, max_length=200)
        return self._postprocess(df, n)

    def _sample_custom(self, n, seed):
        """Generate serialized lines from the fine-tuned model, parse them back with the
        project's own parser (reverse of the exact serializer used in fit), coerce to the
        schema, pad to n. Logs a valid-row yield (like ICL)."""
        set_all_seeds(seed)
        import math

        import torch

        import random

        tok, model = self._tok, self.model
        scheme = self.scheme
        batch = max(1, int(os.environ.get("GREAT_GEN_BATCH", self.params.get("gen_batch", 16))))
        max_oversample = float(self.params.get("max_oversample", 3))
        n_cols = len(self.cols_in_order)
        per_record_tokens = max(48, 14 * n_cols)
        max_new = int(os.environ.get("GREAT_MAX_NEW_TOKENS", min(per_record_tokens, 256)))
        max_attempts = max(1, math.ceil(max_oversample * n / batch))
        first = self.cols_in_order[0]
        temperature = float(self.params.get("temperature", 0.8))

        def _prefix_text() -> str:
            """Scheme-appropriate LINE-START to prime generation (matches the token
            distribution the model saw at position 0). Column-name anchor for
            sentence/keyvalue (value-agnostic); a sampled real first value for compact."""
            if scheme == "sentence":
                return f"The {first} is"
            if scheme == "keyvalue":
                return f"{first}="
            return random.choice(self._compact_starts) if self._compact_starts else ""

        valid: list[dict] = []
        n_parsed = 0
        for _ in range(max_attempts):
            if len(valid) >= n:
                break
            prefix = _prefix_text()
            enc = tok(prefix, return_tensors="pt").to(self._device)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=True, temperature=temperature, top_p=0.95,
                    max_new_tokens=max_new, pad_token_id=tok.eos_token_id,
                    num_return_sequences=batch,
                )
            for seq in out:
                cont = tok.decode(seq[in_len:], skip_special_tokens=True)
                full = prefix + cont            # reattach the primed line-start
                for line in full.splitlines():
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
                f"GReaT(custom scheme={scheme}) produced 0 valid rows; yield=0. "
                "Recorded as a failed cell."
            )
        df = pd.DataFrame(valid)[self.columns]
        if len(df) < n:
            df = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
        return df.iloc[:n].reset_index(drop=True)

    def _coerce(self, row: dict) -> dict | None:
        """Validate + cast a parsed row to the canonical schema (numeric cols must parse
        as numbers). Mirrors ICLGenerator._coerce."""
        out = {}
        for c in self.columns:
            v = row.get(c)
            if c in self.numeric_cols:
                try:
                    f = float(str(v).replace(",", ""))
                except (TypeError, ValueError):
                    return None
                out[c] = int(f) if pd.api.types.is_integer_dtype(self._dtypes[c]) and float(f).is_integer() else f
            else:
                out[c] = str(v)
        return out

    # ------------------------------------------------------------ native post
    def _postprocess(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Coerce GReaT output back to the canonical schema: keep known columns,
        coerce numeric dtypes, drop unparseable rows, realign column order, pad to n."""
        for c in self.columns:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[self.columns].copy()
        for c in self.columns:
            if pd.api.types.is_numeric_dtype(self._dtypes[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=0).reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("GReaT produced 0 parseable rows for this cell.")
        if len(df) < n:
            df = df.sample(n=n, replace=True, random_state=0).reset_index(drop=True)
        return df.iloc[:n].reset_index(drop=True)
