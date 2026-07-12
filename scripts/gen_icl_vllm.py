"""Load-once ICL generation driver (vLLM).

The two-phase runner spawns a fresh subprocess per cell, which reloads the vLLM
engine (~3-4 min) for EACH of the 105 ICL cells -> ~6h of pure reload overhead.
This driver loads the 7B engine ONCE and streams all ICL cells through it, then you
run Phase-B metrics with `scripts/eval_only.py` as usual.

Identical generation logic to the runner path: it uses ICLGenerator (same prompt,
sampling, parser, padding, yield) with the shared engine injected. Writes the same
`generated.csv` + `gen_meta.json` into the same cell dirs, and RESUMES (skips cells
whose csv already matches the config hash + requested n).

Usage (inside the vLLM container, results_dir via config or --results-dir):
  python3 scripts/gen_icl_vllm.py [--results-dir results] [--max-cells N] \
      [--datasets diabetes,adult,student]
Then metrics:
  python3 scripts/eval_only.py --results-dir <same dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import configs, data                       # noqa: E402
from src.generators.icl import ICLGenerator         # noqa: E402


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "transformers", "vllm", "numpy", "pandas"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def _fresh(gm_path: Path, cfg_hash: str, n_gen: int) -> bool:
    if not gm_path.exists():
        return False
    try:
        m = json.loads(gm_path.read_text())
    except Exception:
        return False
    return m.get("config_hash") == cfg_hash and m.get("n_requested") == n_gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results-dir", default=None, help="override cfg paths.results_dir")
    ap.add_argument("--datasets", default=None, help="comma list subset (default: all)")
    ap.add_argument("--max-cells", type=int, default=0, help="stop after N cells (smoke)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_dir = Path(args.results_dir or cfg["paths"]["results_dir"])
    cells_dir = results_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    icl_cfg = next(g for g in cfg["generators"] if g["name"] == "icl")
    icl_params = icl_cfg.get("params", {})
    base_model = icl_cfg.get("base_model")
    all_confs = configs.enumerate_configs(cfg["config"])          # 7 config points
    seeds = cfg["seeds"]
    want_ds = set(args.datasets.split(",")) if args.datasets else None

    # --- data + fixed splits (same cached splits as GReaT/controls) ---
    data.ensure_datasets(cfg["paths"]["data_dir"])
    splits = {}
    for ds in cfg["datasets"]:
        name = ds["name"]
        if want_ds and name not in want_ds:
            continue
        df = data.load_dataset(name, cfg)
        dsc = data._ds_cfg(name, cfg)
        train, holdout = data.fixed_split(df, dsc["target"], name=name, results_dir=results_dir)
        n_gen = min(int(cfg["sample_size"]["n"]), len(train))
        splits[name] = (train, dsc, n_gen)

    # --- build the ICL cell list ---
    cells = []
    for name in splits:
        _, _, n_gen = splits[name]
        for conf in all_confs:
            for seed in seeds:
                cell = cells_dir / f"{name}__icl__{conf['id']}__seed{seed}"
                cells.append((name, conf, seed, n_gen, cell))
    todo = [c for c in cells if not (c[4] / "generated.csv").exists()
            or not _fresh(c[4] / "gen_meta.json", c[1]["hash"], c[3])]
    print(f"[icl-vllm] {len(cells)} ICL cells total, {len(todo)} to generate "
          f"(rest cached).", flush=True)

    # --- load the vLLM engine ONCE ---
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["ICL_BACKEND"] = "vllm"
    from vllm import LLM
    hf_id = base_model if "/" in base_model else f"Qwen/{base_model}"
    print(f"[icl-vllm] loading engine {hf_id} (once) ...", flush=True)
    llm = LLM(
        model=hf_id, dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.90")),
        enforce_eager=os.environ.get("VLLM_EAGER", "1") == "1",  # 0 = CUDA graphs (big GPU)
        max_model_len=int(os.environ.get("VLLM_MAX_LEN", "16384")),
    )
    tok = llm.get_tokenizer()
    print("[icl-vllm] engine ready.", flush=True)

    done = 0
    for i, (name, conf, seed, n_gen, cell) in enumerate(todo, 1):
        cell.mkdir(parents=True, exist_ok=True)
        train, dsc, _ = splits[name]
        meta = {
            "dataset": name, "generator": "icl", "config_id": conf["id"],
            "config_hash": conf["hash"], "seed": seed, "n_requested": n_gen,
            "versions": _versions(),
        }
        try:
            gen = ICLGenerator(icl_params, base_model)
            gen._vllm = llm          # inject the shared engine (no reload)
            gen._tok = tok
            gen.fit(train, seed=seed, config=conf, ds_cfg=dsc)
            t1 = time.time()
            synth = gen.sample(n_gen, seed=seed)
            meta["sample_seconds"] = round(time.time() - t1, 2)
            meta["n_generated"] = len(synth)
            if getattr(gen, "last_yield", None) is not None:
                meta["icl_valid_yield"] = round(float(gen.last_yield), 4)
            (cell / "metrics.json").unlink(missing_ok=True)   # force Phase-B recompute
            synth.to_csv(cell / "generated.csv", index=False)
            meta["status"] = "ok"
            print(f"  [{i}/{len(todo)}] {cell.name}  "
                  f"yield={meta.get('icl_valid_yield')}  {meta['sample_seconds']}s", flush=True)
        except Exception as e:  # noqa: BLE001 — record + keep going
            meta["status"] = "error"
            meta["error"] = f"{type(e).__name__}: {e}"
            meta["traceback"] = traceback.format_exc()
            print(f"  [{i}/{len(todo)}] {cell.name}  ERROR {type(e).__name__}: {e}", flush=True)
        (cell / "gen_meta.json").write_text(json.dumps(meta, indent=2))
        done += 1
        if args.max_cells and done >= args.max_cells:
            print(f"[icl-vllm] stopping after {done} cells (--max-cells).", flush=True)
            break

    print(f"[icl-vllm] done. generated {done} cells. "
          f"Now run: python3 scripts/eval_only.py --results-dir {results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
