#!/usr/bin/env python3
"""Treatment validation: does the proposed protocol (average over multiple
configurations + seeds) actually recover the 'true' generator ranking that a
single-configuration pick misses?

Reference ('ground truth') ranking = rank generators by the grand mean of each
metric over ALL their (config, seed) cells on diabetes. We then Monte-Carlo a
benchmarking 'protocol' that uses k configurations x s seeds per generator and
measure P(it names a generator other than the true best). If the protocol works,
P(wrong) should fall sharply as k grows — i.e., the disease (single-config) is
real and the prescribed treatment (multi-config + CI) cures it.

Run from repo root: /usr/bin/python3 scripts/treatment_validation.py
"""
import csv, glob, json, os, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_gpu")
DS = "diabetes"
METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
GENERATORS = ["ctgan", "tvae", "great", "icl"]
N_MC = 4000
random.seed(20260622)

# vals[metric][generator][config] = list of per-seed metric values
vals = {m: defaultdict(lambda: defaultdict(list)) for m in METRICS}
for cell in sorted(glob.glob(os.path.join(RESULTS, "cells", f"{DS}__*"))):
    mpath = os.path.join(cell, "metrics.json")
    if not os.path.exists(mpath):
        continue
    with open(mpath) as f:
        d = json.load(f)
    gen, cfg = d["generator"], d["config_id"]
    for m in METRICS:
        v = d.get(m)
        if v is not None and v == v:  # not NaN
            vals[m][gen][cfg].append(v)

def best_of(scores, metric):
    """Return the generator with the best score for this metric."""
    if metric == "tstr":
        return max(scores, key=lambda g: scores[g])
    if metric == "c2st_auc":
        return min(scores, key=lambda g: abs(scores[g] - 0.5))
    return min(scores, key=lambda g: scores[g])  # ks/tvd/corr lower better

def grand_score(metric, gen):
    cfgs = vals[metric][gen]
    allv = [v for lst in cfgs.values() for v in lst]
    return sum(allv) / len(allv)

# reference ('true') ranking from the grand mean
true_score = {m: {g: grand_score(m, g) for g in GENERATORS} for m in METRICS}
true_best = {m: best_of(true_score[m], m) for m in METRICS}

def protocol_pick(metric, k, s):
    """One benchmarking run: per generator, sample k configs x s seeds, average."""
    scores = {}
    for g in GENERATORS:
        cfgs = list(vals[metric][g].keys())
        chosen = random.sample(cfgs, min(k, len(cfgs)))
        cell_means = []
        for c in chosen:
            seeds = vals[metric][g][c]
            samp = random.sample(seeds, min(s, len(seeds)))
            cell_means.append(sum(samp) / len(samp))
        scores[g] = sum(cell_means) / len(cell_means)
    return best_of(scores, metric)

def p_wrong(metric, k, s):
    wrong = sum(protocol_pick(metric, k, s) != true_best[metric] for _ in range(N_MC))
    return wrong / N_MC

# protocols: (label, k_configs, s_seeds)
PROTO = [("single (1 cfg, 1 seed)", 1, 1),
         ("1 cfg, 5 seeds", 1, 5),
         ("3 cfgs, 5 seeds", 3, 5),
         ("all cfgs, 5 seeds", 99, 5)]

print(f"\nTreatment validation (diabetes) — P(protocol names the WRONG best generator)")
print(f"reference = grand-mean ranking; true best per metric: "
      + ", ".join(f"{m.split('_')[0]}={true_best[m]}" for m in METRICS) + "\n")
header = f"{'metric':<12}" + "".join(f"{lbl:>22}" for lbl, _, _ in PROTO)
print(header); print("-" * len(header))
out = []
for m in METRICS:
    row = [p_wrong(m, k, s) for _, k, s in PROTO]
    print(f"{m:<12}" + "".join(f"{x:>22.3f}" for x in row))
    out.append((DS, m, true_best[m], *[round(x, 4) for x in row]))

out_path = os.path.join(RESULTS, "analysis", "treatment_validation.csv")
with open(out_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "metric", "true_best",
                "p_wrong_1cfg_1seed", "p_wrong_1cfg_5seed",
                "p_wrong_3cfg_5seed", "p_wrong_allcfg_5seed"])
    w.writerows(out)
print(f"\nwrote {out_path}")
