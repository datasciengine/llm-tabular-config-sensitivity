"""Phase-B-only: compute the 5 metrics for cached generated.csv cells WITHOUT any
generation. Runs locally in the neutral env (system python3), so no GPU / no venvs /
no downloads (datasets + splits are already on disk).

Use after copying the GPU's results/ back: it fills in metrics.json for every cell
that has a generated.csv but no metrics.json yet, isolating per-cell failures.

    /usr/bin/python3 scripts/eval_only.py --results-dir results-gpu
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src import data, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_dir = Path(args.results_dir)
    cells_dir = results_dir / "cells"

    # Prepare splits once per dataset (uses cached results/splits if present; reads
    # raw data from ./data — both already local, no download).
    data.ensure_datasets(cfg["paths"]["data_dir"])
    splits = {}
    for ds in cfg["datasets"]:
        name = ds["name"]
        df = data.load_dataset(name, cfg)
        dsc = data._ds_cfg(name, cfg)
        train, holdout = data.fixed_split(df, dsc["target"], name=name, results_dir=results_dir)
        splits[name] = (train, holdout, dsc)

    done = skipped = failed = 0
    for cell in sorted(cells_dir.glob("*/")):
        gpath = cell / "generated.csv"
        mpath = cell / "metrics.json"
        if not gpath.exists():
            continue
        if mpath.exists():
            skipped += 1
            continue
        # name = {dataset}__{generator}__{config_id}__seed{n}  (config_id has no '__')
        parts = cell.name.split("__")
        name, gen, cid, seedpart = parts[0], parts[1], parts[2], parts[3]
        seed = int(seedpart.replace("seed", ""))
        if name not in splits:
            continue
        train, holdout, dsc = splits[name]
        try:
            synth = pd.read_csv(gpath)
            m = metrics.compute_all(train, synth, holdout, dsc, seed=seed)
            m.update({"dataset": name, "generator": gen, "config_id": cid, "seed": seed})
            mpath.write_text(json.dumps(m, indent=2))
            done += 1
            print(f"  ok   {cell.name}  ks={m['ks_marginal']:.3f} c2st={m['c2st_auc']:.3f}")
        except Exception as e:
            (cell / "metrics_error.json").write_text(json.dumps(
                {"status": "metrics_error", "error": f"{type(e).__name__}: {e}",
                 "dataset": name, "generator": gen, "config_id": cid, "seed": seed}, indent=2))
            failed += 1
            print(f"  ERR  {cell.name}  {type(e).__name__}: {e}")

    print(f"\n[eval_only] computed={done}  already_had_metrics={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
