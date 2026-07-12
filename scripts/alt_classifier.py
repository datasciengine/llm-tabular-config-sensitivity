#!/usr/bin/env python3
"""classifier-sensitivity check for the two learned metrics.

C2ST uses LogisticRegression and TSTR uses HistGradientBoosting in the frozen
protocol. One may ask whether the configuration-sensitivity conclusions hinge
on those single classifier choices. This recomputes C2ST-AUC and TSTR-AUC on every
diabetes cell under several alternative learners and reports, per generator:
  - the mean metric (does the generator ordering hold?), and
  - the across-configuration spread std (the CSI ingredient: is 'config moves the
    score' still true under a different classifier?).

If the qualitative story (LLM generators show larger across-config spread than the
config-invariant controls; generator ordering is stable) survives every classifier,
the conclusion is not an artifact of the two frozen learners.

    /usr/bin/python3 scripts/alt_classifier.py --results-dir results_reseed --config config_reseed.yaml
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src import data
from src.metrics import _build_preprocessor, _types, _align_synth_dtypes

GENERATORS = ["ctgan", "tvae", "great", "icl"]

# name -> factory (fresh estimator each call)
C2ST_CLFS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    "gb": lambda: GradientBoostingClassifier(random_state=0),
}
TSTR_CLFS = {
    "hgb": None,  # default; imported lazily below
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
}


def c2st(real, synth, dsc, clf_factory, seed):
    numeric, categorical, _ = _types(real, dsc)
    synth = synth[real.columns].copy()
    X = pd.concat([real, synth], ignore_index=True)
    y = np.r_[np.zeros(len(real)), np.ones(len(synth))]
    pipe = Pipeline([("prep", _build_preprocessor(numeric, categorical)),
                     ("clf", clf_factory())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")))


def tstr(real_train, synth, holdout, dsc, clf_factory):
    target = dsc["target"]
    drop = set(dsc.get("drop_for_tstr", [])) | {target}
    feats = [c for c in real_train.columns if c not in drop]
    numeric = [c for c in feats if c in _types(real_train, dsc)[0]]
    categorical = [c for c in feats if c in _types(real_train, dsc)[1]]
    pos = dsc.get("positive_class", 1)
    ytr = (synth[target].astype(str) == str(pos)).astype(int).values
    yte = (holdout[target].astype(str) == str(pos)).astype(int).values
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    pipe = Pipeline([("prep", _build_preprocessor(numeric, categorical)),
                     ("clf", clf_factory())])
    pipe.fit(synth[feats], ytr)
    proba = pipe.predict_proba(holdout[feats])[:, 1]
    return float(roc_auc_score(yte, proba))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_reseed.yaml"))
    ap.add_argument("--results-dir", default="results_reseed")
    ap.add_argument("--dataset", default="diabetes")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    TSTR_CLFS["hgb"] = lambda: HistGradientBoostingClassifier(random_state=0)

    cfg = yaml.safe_load(open(args.config))
    results_dir = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    DS = args.dataset

    data.ensure_datasets(cfg["paths"]["data_dir"])
    df = data.load_dataset(DS, cfg)
    dsc = data._ds_cfg(DS, cfg)
    from pathlib import Path
    train, holdout = data.fixed_split(df, dsc["target"], name=DS, results_dir=Path(results_dir))

    # per (metric-clf) -> gen -> config -> list-over-seeds
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cells = sorted(glob.glob(os.path.join(results_dir, "cells", f"{DS}__*")))
    print(f"[alt_classifier] {len(cells)} cells\n")
    for cell in cells:
        parts = os.path.basename(cell).split("__")
        gen, cfg_id, seed = parts[1], parts[2], int(parts[3].replace("seed", ""))
        if gen not in GENERATORS:
            continue
        gpath = os.path.join(cell, "generated.csv")
        if not os.path.exists(gpath):
            continue
        synth = _align_synth_dtypes(train, pd.read_csv(gpath))
        if len(synth) < 10:
            continue
        for cname, fac in C2ST_CLFS.items():
            store[f"c2st_{cname}"][gen][cfg_id].append(c2st(train, synth, dsc, fac, seed))
        for cname, fac in TSTR_CLFS.items():
            store[f"tstr_{cname}"][gen][cfg_id].append(tstr(train, synth, holdout, dsc, fac))

    def gen_mean(metric, gen):
        xs = [v for c in store[metric][gen] for v in store[metric][gen][c] if v == v]
        return sum(xs) / len(xs) if xs else float("nan")

    def config_spread(metric, gen):
        """std of per-config means (the CSI ingredient). For a config-invariant control
        (a single config), fall back to the across-SEED std within that config — the
        seed-only null floor the LLM spread must beat."""
        cms = []
        for c in store[metric][gen]:
            xs = [v for v in store[metric][gen][c] if v == v]
            if xs:
                cms.append(sum(xs) / len(xs))
        if len(cms) > 1:
            return float(np.std(cms))
        # single config -> seed floor
        for c in store[metric][gen]:
            xs = [v for v in store[metric][gen][c] if v == v]
            if len(xs) > 1:
                return float(np.std(xs))
        return float("nan")

    out = []
    gens = [g for g in GENERATORS if any(g in store[m] and store[m][g] for m in store)]
    for metric in list(C2ST_CLFS and [f"c2st_{c}" for c in C2ST_CLFS]) + [f"tstr_{c}" for c in TSTR_CLFS]:
        print(f"--- {metric} ---")
        print(f"{'generator':<8} {'mean':>8} {'config-spread(std)':>20}")
        for g in gens:
            gm, sp = gen_mean(metric, g), config_spread(metric, g)
            print(f"{g:<8} {gm:>8.3f} {sp:>20.4f}")
            out.append((DS, metric, g, round(gm, 4), round(sp, 4) if sp == sp else ""))
        print()

    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
    out_path = os.path.join(results_dir, "analysis", "alt_classifier.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "metric_clf", "generator", "mean", "config_spread_std"])
        w.writerows(out)
    print(f"wrote {out_path}")
    print("\nRead: within each learned metric, compare the config-spread(std) of the LLM "
          "generators (great, icl) against the config-invariant controls (ctgan, tvae). "
          "If LLM spread > control spread under EVERY classifier, config-sensitivity is "
          "not an artifact of the frozen LogReg/HGB choice.")


if __name__ == "__main__":
    main()
