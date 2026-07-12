#!/usr/bin/env python3
"""Dedup-reeval: recompute ICL fidelity on diabetes AFTER removing rows that are
verbatim copies of a real training row. Tests whether ICL's apparent fidelity
advantage is an artifact of memorization (synth ~= real subset).

For each ICL diabetes cell we compute the 5 frozen metrics on (a) the full generated
set and (b) the generated set with verbatim training-row copies removed. If the
advantage is copying, (b) should be markedly worse (KS/TVD/corr up, C2ST up, TSTR down).

Run from repo root: /usr/bin/python3 scripts/dedup_reeval.py
"""
import csv, glob, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from src.metrics import compute_all

RESULTS = os.path.join(ROOT, "results_gpu")
DS = "diabetes"
DS_CFG = {"target": "Outcome", "positive_class": 1, "drop_for_tstr": []}
METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]

real_train = pd.read_csv(os.path.join(RESULTS, "splits", DS, "train.csv"))
real_holdout = pd.read_csv(os.path.join(RESULTS, "splits", DS, "holdout.csv"))

def norm_key(df):
    """Row-key set after numeric normalization (27.0==27, 0.10==0.1)."""
    def nv(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else repr(round(f, 6))
        except (ValueError, TypeError):
            return str(v)
    return {tuple(nv(v) for v in row) for row in df.itertuples(index=False, name=None)}

train_keys = norm_key(real_train)

def is_copy_mask(synth):
    def nv(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else repr(round(f, 6))
        except (ValueError, TypeError):
            return str(v)
    return [tuple(nv(v) for v in row) in train_keys
            for row in synth.itertuples(index=False, name=None)]

def seed_of(name):
    return int(name.split("seed")[-1])

def ser_of(config):
    if "serialization=" in config:
        return config.split("serialization=")[1]
    return "sentence"  # baseline / column_order / numeric_format all keep sentence

agg_full = defaultdict(lambda: defaultdict(list))
agg_dedup = defaultdict(lambda: defaultdict(list))
copy_frac = defaultdict(list)

cells = sorted(glob.glob(os.path.join(RESULTS, "cells", f"{DS}__icl__*")))
for cell in cells:
    name = os.path.basename(cell)
    config = name.split("__")[2]
    seed = seed_of(name)
    ser = ser_of(config)
    gpath = os.path.join(cell, "generated.csv")
    if not os.path.exists(gpath):
        continue
    synth = pd.read_csv(gpath)
    mask = is_copy_mask(synth)
    n = len(synth)
    n_copy = sum(mask)
    copy_frac[ser].append(n_copy / n)
    dedup = synth[[not m for m in mask]].reset_index(drop=True)

    full_m = compute_all(real_train, synth, real_holdout, DS_CFG, seed=seed)
    # dedup may be small / single-class for TSTR; compute_all handles NaN gracefully
    dd_m = compute_all(real_train, dedup, real_holdout, DS_CFG, seed=seed) if len(dedup) >= 10 else {k: float("nan") for k in METRICS}

    for m in METRICS:
        agg_full[ser][m].append(full_m[m])
        agg_dedup[ser][m].append(dd_m[m])

def mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")

out = []
print(f"\nDiabetes ICL — fidelity FULL vs DEDUP (verbatim train-copies removed)\n")
print(f"{'serialization':<12} {'copy%':>6} {'metric':<10} {'full':>8} {'dedup':>8} {'Δ (worse?)':>12}")
print("-" * 64)
for ser in ["sentence", "keyvalue", "compact"]:
    cf = 100 * mean(copy_frac[ser])
    for m in METRICS:
        f = mean(agg_full[ser][m])
        d = mean(agg_dedup[ser][m])
        delta = d - f
        # direction: KS/TVD/corr/C2ST worse=up; TSTR worse=down
        worse = (delta > 0) if m != "tstr" else (delta < 0)
        flag = "  worse" if worse else "  ~same/better"
        print(f"{ser:<12} {cf:>5.1f}% {m:<10} {f:>8.3f} {d:>8.3f} {delta:>+12.3f}{flag}")
        out.append((DS, ser, round(cf, 2), m, round(f, 4), round(d, 4), round(delta, 4)))
    print()

out_path = os.path.join(RESULTS, "analysis", "dedup_reeval.csv")
with open(out_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "serialization", "copy_pct", "metric", "full", "dedup", "delta"])
    w.writerows(out)
print(f"wrote {out_path}")
