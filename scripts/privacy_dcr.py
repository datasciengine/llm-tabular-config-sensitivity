#!/usr/bin/env python3
"""standard privacy metrics — DCR and NNDR — with a real-data baseline.

The paper observes memorization but reported no privacy metric. This adds the two
most common ones, computed against the real TRAIN set in a shared numeric embedding
(one-hot categoricals + standardized numerics, fit on real):

  DCR  (Distance to Closest Record): for each synthetic row, the Euclidean distance
       to its nearest real-train row. Verbatim copies have DCR = 0. We report the
       median and the 5th percentile (the risky tail), plus the exact-copy fraction.
  NNDR (Nearest-Neighbour Distance Ratio): dist(1st NN) / dist(2nd NN) in real-train.
       Near 0 = the synthetic row sits almost on a single real record (identifying);
       near 1 = it lies between records (safer). We report the median.

Everything is CALIBRATED against a real baseline: the holdout rows' DCR/NNDR to the
train set. A privacy-respecting generator should not produce DCR *smaller* than this
real-vs-real floor. This turns "privacy" from an unquantified claim into numbers.

    /usr/bin/python3 scripts/privacy_dcr.py --results-dir results_reseed --config config_reseed.yaml
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

from src import data
from src.metrics import _build_preprocessor, _types, _align_synth_dtypes

GENERATORS = ["ctgan", "tvae", "great", "icl"]


def ser_of(config_id):
    if "serialization=" in config_id:
        return config_id.split("serialization=")[1].split("__")[0]
    return "sentence"


def dcr_nndr(embed_real, embed_query, nn2):
    """Return (dcr array, nndr array) for query rows against the real embedding."""
    dist, _ = nn2.kneighbors(embed_query, n_neighbors=2)
    d1 = dist[:, 0]
    d2 = dist[:, 1]
    nndr = np.where(d2 > 0, d1 / d2, 0.0)
    return d1, nndr


def summarize(dcr, nndr):
    return {
        "dcr_median": float(np.median(dcr)),
        "dcr_p05": float(np.percentile(dcr, 5)),
        "exact_copy_frac": float(np.mean(dcr <= 1e-9)),
        "nndr_median": float(np.median(nndr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_reseed.yaml"))
    ap.add_argument("--results-dir", default="results_reseed")
    ap.add_argument("--dataset", default="diabetes")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_dir = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    DS = args.dataset

    data.ensure_datasets(cfg["paths"]["data_dir"])
    df = data.load_dataset(DS, cfg)
    dsc = data._ds_cfg(DS, cfg)
    from pathlib import Path
    train, holdout = data.fixed_split(df, dsc["target"], name=DS, results_dir=Path(results_dir))

    numeric, categorical, _ = _types(train, dsc)
    prep = _build_preprocessor(numeric, categorical).fit(train)
    E_train = prep.transform(train)
    nn2 = NearestNeighbors(n_neighbors=2).fit(E_train)

    # real baseline: holdout vs train (a genuinely disjoint real sample)
    hb_dcr, hb_nndr = dcr_nndr(E_train, prep.transform(holdout), nn2)
    base = summarize(hb_dcr, hb_nndr)
    print(f"\nPrivacy (diabetes) — DCR / NNDR vs real train, in one-hot+standardized space")
    print(f"REAL BASELINE (holdout→train): dcr_median={base['dcr_median']:.3f} "
          f"dcr_p05={base['dcr_p05']:.3f} nndr_median={base['nndr_median']:.3f} "
          f"exact_copy={100*base['exact_copy_frac']:.1f}%")
    print("(synthetic DCR below the baseline median = closer to real records than a real "
          "holdout is → a privacy red flag.)\n")

    # aggregate per (generator, serialization-family)
    agg = defaultdict(lambda: defaultdict(list))  # (gen,ser) -> stat -> [per-cell values]
    for cell in sorted(glob.glob(os.path.join(results_dir, "cells", f"{DS}__*"))):
        parts = os.path.basename(cell).split("__")
        gen, cfg_id = parts[1], parts[2]
        if gen not in GENERATORS:
            continue
        gpath = os.path.join(cell, "generated.csv")
        if not os.path.exists(gpath):
            continue
        synth = _align_synth_dtypes(train, pd.read_csv(gpath))
        if len(synth) < 5:
            continue
        try:
            E = prep.transform(synth[train.columns])
        except Exception:
            continue
        dcr, nndr = dcr_nndr(E_train, E, nn2)
        s = summarize(dcr, nndr)
        ser = ser_of(cfg_id)
        for stat, v in s.items():
            agg[(gen, ser)][stat].append(v)
            agg[(gen, "ALL")][stat].append(v)

    STATS = ["dcr_median", "dcr_p05", "exact_copy_frac", "nndr_median"]
    out = [("BASELINE_holdout", "-", *[round(base[s], 4) for s in STATS])]
    print(f"{'generator':<8} {'serialization':<12} {'dcr_med':>8} {'dcr_p05':>8} "
          f"{'copy%':>7} {'nndr_med':>9}")
    print("-" * 56)
    for (gen, ser) in sorted(agg):
        m = {s: float(np.mean(agg[(gen, ser)][s])) for s in STATS}
        print(f"{gen:<8} {ser:<12} {m['dcr_median']:>8.3f} {m['dcr_p05']:>8.3f} "
              f"{100*m['exact_copy_frac']:>6.1f}% {m['nndr_median']:>9.3f}")
        out.append((gen, ser, *[round(m[s], 4) for s in STATS]))

    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
    out_path = os.path.join(results_dir, "analysis", "privacy_dcr.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "serialization", "dcr_median", "dcr_p05",
                    "exact_copy_frac", "nndr_median"])
        w.writerows(out)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
