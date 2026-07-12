#!/usr/bin/env python3
"""#7 FAST: same analysis as dedup_fixed_n.py but parallel over cells and reusing the
already-computed full metrics from metrics.json (so only dedup + size-matched are
recomputed). ~6 workers -> minutes instead of ~half an hour.

    /usr/bin/python3 scripts/dedup_fixed_n_fast.py --results-dir results_reseed --config config_reseed.yaml
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
MIN_DEDUP = 10

# per-worker globals (set by _init)
_TRAIN = _HOLD = _DSC = _KEYS = None


def _nv(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else repr(round(f, 6))
    except (ValueError, TypeError):
        return str(v)


def _keyset(df):
    return {tuple(_nv(v) for v in row) for row in df.itertuples(index=False, name=None)}


def _init(results_dir, config_path, dataset):
    global _TRAIN, _HOLD, _DSC, _KEYS
    import warnings
    warnings.filterwarnings("ignore")
    import yaml
    from pathlib import Path
    from src import data
    cfg = yaml.safe_load(open(config_path))
    df = data.load_dataset(dataset, cfg)
    _DSC = data._ds_cfg(dataset, cfg)
    _TRAIN, _HOLD = data.fixed_split(df, _DSC["target"], name=dataset, results_dir=Path(results_dir))
    _KEYS = _keyset(_TRAIN)


def _work(args):
    """One cell: return (ser, {phase: {metric: val}}, copy_frac)."""
    cell, config_id, seed = args
    from src import metrics
    gpath = os.path.join(cell, "generated.csv")
    synth = pd.read_csv(gpath)
    n = len(synth)
    if n == 0:
        return None
    mask = np.array([tuple(_nv(v) for v in row) in _KEYS
                     for row in synth.itertuples(index=False, name=None)])
    n_copy = int(mask.sum())
    n_dedup = n - n_copy
    ser = config_id.split("serialization=")[1].split("__")[0] if "serialization=" in config_id else "sentence"

    # full: read cached metrics.json (already computed) -> free
    mpath = os.path.join(cell, "metrics.json")
    full_m = {m: json.load(open(mpath)).get(m) for m in METRICS} if os.path.exists(mpath) \
        else metrics.compute_all(_TRAIN, synth, _HOLD, _DSC, seed=seed)

    if n_dedup >= MIN_DEDUP:
        dedup = synth[~mask].reset_index(drop=True)
        dedup_m = metrics.compute_all(_TRAIN, dedup, _HOLD, _DSC, seed=seed)
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=n_dedup, replace=False)
        sized_m = metrics.compute_all(_TRAIN, synth.iloc[idx].reset_index(drop=True), _HOLD, _DSC, seed=seed)
    else:
        dedup_m = {m: float("nan") for m in METRICS}
        sized_m = {m: float("nan") for m in METRICS}
    return ser, {"full": full_m, "dedup": dedup_m, "sized": sized_m}, n_copy / n


def mean_ci(xs):
    xs = [x for x in xs if x is not None and x == x]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.959963984540054 * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_reseed.yaml"))
    ap.add_argument("--results-dir", default="results_reseed")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-name", default="dedup_fixed_n")
    args = ap.parse_args()

    results_dir = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    DS = args.dataset
    cells = sorted(glob.glob(os.path.join(results_dir, "cells", f"{DS}__icl__*")))
    tasks = []
    for cell in cells:
        parts = os.path.basename(cell).split("__")
        if os.path.exists(os.path.join(cell, "generated.csv")):
            tasks.append((cell, parts[2], int(parts[3].replace("seed", ""))))
    print(f"[dedup_fast] {len(tasks)} ICL cells, {args.workers} workers")

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    copy_fracs = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init,
                             initargs=(results_dir, args.config, DS)) as ex:
        for res in ex.map(_work, tasks):
            if res is None:
                continue
            ser, phases, cf = res
            copy_fracs[ser].append(cf)
            for phase in ("full", "dedup", "sized"):
                for m in METRICS:
                    agg[ser][phase][m].append(phases[phase][m])

    out = []
    print(f"{'ser':<10} {'copy%':>6} {'metric':<10} {'full':>16} {'dedup':>16} {'sizematch':>16} {'dedup-sized':>14}")
    print("-" * 92)
    for ser in ["sentence", "keyvalue", "compact"]:
        if ser not in agg:
            continue
        cf = 100 * sum(copy_fracs[ser]) / len(copy_fracs[ser])
        for m in METRICS:
            fm, fci = mean_ci(agg[ser]["full"][m])
            dm, dci = mean_ci(agg[ser]["dedup"][m])
            sm, sci = mean_ci(agg[ser]["sized"][m])
            eff = dm - sm if (dm == dm and sm == sm) else float("nan")
            print(f"{ser:<10} {cf:>5.1f}% {m:<10} {fm:>8.3f}±{fci:<6.3f} {dm:>8.3f}±{dci:<6.3f} "
                  f"{sm:>8.3f}±{sci:<6.3f} {eff:>+14.3f}")
            out.append((DS, ser, round(cf, 2), m, round(fm, 4), round(fci, 4),
                        round(dm, 4), round(dci, 4), round(sm, 4), round(sci, 4),
                        round(eff, 4) if eff == eff else ""))
        print()

    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
    out_path = os.path.join(results_dir, "analysis", f"{args.out_name}.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "serialization", "copy_pct", "metric", "full_mean", "full_ci95",
                    "dedup_mean", "dedup_ci95", "sizematched_mean", "sizematched_ci95",
                    "dedup_minus_sizematched"])
        w.writerows(out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
