#!/usr/bin/env python3
"""Measure exact-copy / memorization rate of generated rows vs. the real train split.

Motivation: the weak 0.5B ICL model may copy its few-shot example rows verbatim,
which would artificially inflate fidelity (synth ~= real subset) and confound the
configuration-sensitivity story. This quantifies that, per generator, on diabetes
(the only cell where ICL is populated), with GReaT/CTGAN/TVAE as comparison baselines.

Run: /usr/bin/python3 scripts/measure_copying.py
"""
import csv, glob, os, re, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_gpu")
DATASET = "diabetes"

def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [tuple(x.strip() for x in row) for row in r]
    return header, rows

def norm_num(v):
    """Normalize a numeric string so 27.0 == 27, 0.10 == 0.1; leave non-numeric as-is."""
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
        return repr(round(f, 6))
    except (ValueError, TypeError):
        return v

def keyset(header, rows):
    """Set of row-keys after numeric normalization, so format-only diffs don't hide copies."""
    s = set()
    for row in rows:
        if len(row) != len(header):
            continue
        s.add(tuple(norm_num(v) for v in row))
    return s

# real train rows
train_path = os.path.join(RESULTS, "splits", DATASET, "train.csv")
th, trows = read_csv(train_path)
train_keys = keyset(th, trows)
print(f"train rows: {len(trows)} (unique keys: {len(train_keys)})\n")

# aggregate per (generator, serialization-family)
agg = defaultdict(list)  # key -> list of copy-fractions per cell

cells = sorted(glob.glob(os.path.join(RESULTS, "cells", f"{DATASET}__*")))
for cell in cells:
    name = os.path.basename(cell)
    parts = name.split("__")
    gen = parts[1]
    config = parts[2] if len(parts) > 2 else "baseline"
    gpath = os.path.join(cell, "generated.csv")
    if not os.path.exists(gpath):
        continue
    gh, grows = read_csv(gpath)
    if not grows:
        continue
    gkeys_list = [tuple(norm_num(v) for v in row) for row in grows if len(row) == len(gh)]
    n = len(gkeys_list)
    if n == 0:
        continue
    copied = sum(1 for k in gkeys_list if k in train_keys)
    frac = copied / n
    # serialization family from config id
    if "serialization=" in config:
        ser = config.split("serialization=")[1].split("__")[0]
    else:
        ser = "sentence(baseline-axis)"  # baseline / column_order / numeric_format keep sentence
    agg[(gen, ser)].append(frac)
    agg[(gen, "ALL")].append(frac)

out_rows = []
print(f"{'generator':<8} {'serialization':<24} {'cells':>5} {'mean copy%':>11} {'max copy%':>10}")
print("-" * 62)
for (gen, ser) in sorted(agg):
    fr = agg[(gen, ser)]
    mean = 100 * sum(fr) / len(fr)
    mx = 100 * max(fr)
    print(f"{gen:<8} {ser:<24} {len(fr):>5} {mean:>10.1f}% {mx:>9.1f}%")
    out_rows.append((DATASET, gen, ser, len(fr), round(mean, 2), round(mx, 2)))

out_path = os.path.join(RESULTS, "analysis", "copy_rate.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "generator", "serialization", "n_cells", "mean_copy_pct", "max_copy_pct"])
    w.writerows(out_rows)
print(f"\nwrote {out_path}")
