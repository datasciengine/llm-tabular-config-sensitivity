#!/usr/bin/env python3
"""dedup fidelity re-eval with the SAMPLE-SIZE confound removed.

A naive dedup compares full (n rows) vs dedup (n - n_copy rows); but the dedup set is *smaller*, and several metrics (esp. C2ST AUC and KS) drift
with n — so "dedup looks worse" could be a pure size artifact, not evidence that the
apparent fidelity was memorization.

This script isolates the two effects. For every ICL cell we compute the 5 metrics on:
  (a) FULL           — all n generated rows (copies retained, natural rate)
  (b) DEDUP          — verbatim train-copies removed  (n_dedup rows)
  (c) SIZE-MATCHED   — a RANDOM subsample of the FULL set down to n_dedup rows
                       (copies retained at natural rate, but same n as DEDUP)

The confound-free contrast is DEDUP vs SIZE-MATCHED: both have n_dedup rows, so any
difference is due to *removing copies*, not to having fewer rows. (FULL vs DEDUP is
still reported for continuity, but it is the confounded comparison.)

Aggregated over seeds within each serialization family, reported as mean ± 95% CI.

    /usr/bin/python3 scripts/dedup_fixed_n.py --results-dir results_reseed \
        --config config_reseed.yaml
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import yaml

from src import data, metrics

METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
MIN_DEDUP = 10  # below this the dedup/size-matched metrics are unreliable -> NaN


def norm_key_set(df):
    """Row-key set after numeric normalization (27.0==27, 0.10==0.1)."""
    def nv(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else repr(round(f, 6))
        except (ValueError, TypeError):
            return str(v)
    return {tuple(nv(v) for v in row) for row in df.itertuples(index=False, name=None)}


def copy_mask(synth, train_keys):
    def nv(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else repr(round(f, 6))
        except (ValueError, TypeError):
            return str(v)
    return np.array([tuple(nv(v) for v in row) in train_keys
                     for row in synth.itertuples(index=False, name=None)])


def ser_of(config_id):
    if "serialization=" in config_id:
        return config_id.split("serialization=")[1].split("__")[0]
    return "sentence"  # baseline / column_order / numeric_format all keep sentence


def mean_ci(xs, level=0.95):
    xs = [x for x in xs if x == x]  # drop NaN
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    # normal approx (n>=5 per family here); half-width of the 95% CI
    z = 1.959963984540054
    return m, z * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_reseed.yaml"))
    ap.add_argument("--results-dir", default="results_reseed")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--out-name", default="dedup_fixed_n")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_dir = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    DS = args.dataset

    # splits (uses cached results_dir/splits if present; else reads ./data)
    data.ensure_datasets(cfg["paths"]["data_dir"])
    df = data.load_dataset(DS, cfg)
    dsc = data._ds_cfg(DS, cfg)
    from pathlib import Path
    train, holdout = data.fixed_split(df, dsc["target"], name=DS, results_dir=Path(results_dir))
    train_keys = norm_key_set(train)

    # per (serialization, phase, metric) -> list over seeds
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # ser -> phase -> metric -> [vals]
    copy_fracs = defaultdict(list)

    cells = sorted(glob.glob(os.path.join(results_dir, "cells", f"{DS}__icl__*")))
    if not cells:
        print(f"[dedup_fixed_n] no ICL cells under {results_dir}/cells/{DS}__icl__* — nothing to do.")
        return
    print(f"[dedup_fixed_n] {len(cells)} ICL cells; train rows={len(train)} "
          f"(unique keys={len(train_keys)})\n")

    for cell in cells:
        name = os.path.basename(cell)
        parts = name.split("__")
        config_id, seed = parts[2], int(parts[3].replace("seed", ""))
        ser = ser_of(config_id)
        gpath = os.path.join(cell, "generated.csv")
        if not os.path.exists(gpath):
            continue
        synth = pd.read_csv(gpath)
        n = len(synth)
        if n == 0:
            continue
        mask = copy_mask(synth, train_keys)
        n_copy = int(mask.sum())
        n_dedup = n - n_copy
        copy_fracs[ser].append(n_copy / n)

        # full metrics are already in metrics.json (computed by eval_only/runner) — reuse
        # them instead of recomputing (TSTR/C2ST are the expensive part). Falls back to a
        # fresh compute only if the cell has no metrics.json.
        mpath = os.path.join(cell, "metrics.json")
        if os.path.exists(mpath):
            _mj = json.load(open(mpath))
            full_m = {m: _mj.get(m) for m in METRICS}
        else:
            full_m = metrics.compute_all(train, synth, holdout, dsc, seed=seed)

        if n_dedup >= MIN_DEDUP:
            dedup = synth[~mask].reset_index(drop=True)
            dedup_m = metrics.compute_all(train, dedup, holdout, dsc, seed=seed)
            # size-matched: random subsample of FULL to n_dedup rows (seeded, reproducible)
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, size=n_dedup, replace=False)
            sized = synth.iloc[idx].reset_index(drop=True)
            sized_m = metrics.compute_all(train, sized, holdout, dsc, seed=seed)
        else:
            dedup_m = {k: float("nan") for k in METRICS}
            sized_m = {k: float("nan") for k in METRICS}

        for m in METRICS:
            agg[ser]["full"][m].append(full_m[m])
            agg[ser]["dedup"][m].append(dedup_m[m])
            agg[ser]["sized"][m].append(sized_m[m])

    # report
    out = []
    print(f"{'ser':<10} {'copy%':>6} {'metric':<10} "
          f"{'full':>16} {'dedup':>16} {'sizematch':>16} {'dedup−sizematch':>18}")
    print("-" * 96)
    for ser in ["sentence", "keyvalue", "compact"]:
        if ser not in agg:
            continue
        cf = 100 * (sum(copy_fracs[ser]) / len(copy_fracs[ser]))
        for m in METRICS:
            fm, fci = mean_ci(agg[ser]["full"][m])
            dm, dci = mean_ci(agg[ser]["dedup"][m])
            sm, sci = mean_ci(agg[ser]["sized"][m])
            # confound-free effect of removing copies (both at n_dedup)
            eff = dm - sm if (dm == dm and sm == sm) else float("nan")
            print(f"{ser:<10} {cf:>5.1f}% {m:<10} "
                  f"{fm:>8.3f}±{fci:<6.3f} {dm:>8.3f}±{dci:<6.3f} "
                  f"{sm:>8.3f}±{sci:<6.3f} {eff:>+18.3f}")
            out.append((DS, ser, round(cf, 2), m,
                        round(fm, 4), round(fci, 4),
                        round(dm, 4), round(dci, 4),
                        round(sm, 4), round(sci, 4),
                        round(eff, 4) if eff == eff else ""))
        print()

    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
    out_path = os.path.join(results_dir, "analysis", f"{args.out_name}.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "serialization", "copy_pct", "metric",
                    "full_mean", "full_ci95", "dedup_mean", "dedup_ci95",
                    "sizematched_mean", "sizematched_ci95", "dedup_minus_sizematched"])
        w.writerows(out)
    print(f"wrote {out_path}")
    print("\nRead: 'dedup−sizematch' is the copy-removal effect with n held fixed. "
          "If it is ~0 while full≠dedup, the dedup shift was a sample-size artifact; "
          "if it is large (KS/TVD/corr up, C2ST toward 0.5→up, TSTR down), removing "
          "memorized rows genuinely degrades fidelity → the fidelity was partly copying.")


if __name__ == "__main__":
    main()
