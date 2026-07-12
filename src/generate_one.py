"""Subprocess generation entrypoint — runs INSIDE a generator's own env.

Usage:
  python -m src.generate_one --config CFG --results-dir DIR --dataset D \
      --generator G --config-id C --seed N --n K --cell CELLDIR

Reads the cached split, builds the generator, fits, samples n rows, and writes
`generated.csv` + `gen_meta.json` into the cell dir. Exceptions are caught and
recorded as status='error' (exit 0) so the runner's grid keeps going.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from . import configs, data
from .generators.base import make_generator


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "transformers", "sdv", "numpy", "pandas", "be_great"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cell", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cell = Path(args.cell)
    cell.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    gcfg = next(g for g in cfg["generators"] if g["name"] == args.generator)
    conf = next(c for c in configs.enumerate_configs(cfg["config"]) if c["id"] == args.config_id)
    ds_cfg = data._ds_cfg(args.dataset, cfg)

    train = pd.read_csv(results_dir / "splits" / args.dataset / "train.csv")

    meta = {
        "dataset": args.dataset, "generator": args.generator,
        "config_id": args.config_id, "config_hash": conf["hash"], "seed": args.seed,
        "n_requested": args.n, "versions": _versions(),
    }

    try:
        # GReaT fine-tunes on a fixed subsample (the project spec sec 12.1).
        fit_df = train
        if gcfg["kind"] == "llm_finetune":
            max_rows = int(cfg["sample_size"].get("great_finetune_max_rows", 5000))
            if len(train) > max_rows:
                fit_df = train.sample(n=max_rows,
                                      random_state=int(cfg["finetune_subsample_seed"]))
            meta["finetune_rows"] = len(fit_df)

        t0 = time.time()
        gen = make_generator(gcfg)
        gen.fit(fit_df, seed=args.seed, config=conf, ds_cfg=ds_cfg)
        meta["fit_seconds"] = round(time.time() - t0, 2)

        t1 = time.time()
        synth = gen.sample(args.n, seed=args.seed)
        meta["sample_seconds"] = round(time.time() - t1, 2)
        meta["n_generated"] = len(synth)
        if getattr(gen, "last_yield", None) is not None:
            meta["icl_valid_yield"] = round(float(gen.last_yield), 4)

        synth.to_csv(cell / "generated.csv", index=False)
        meta["status"] = "ok"
    except Exception as e:  # noqa: BLE001 — record and continue the grid
        meta["status"] = "error"
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc()

    (cell / "gen_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
