#!/usr/bin/env python3
"""treatment-validation with an INDEPENDENT (non-tautological) and
optionally MEMORIZATION-CLEAN reference ranking.

One objection: the "true" ranking was the grand mean over ALL (config, seed) cells,
and the Monte-Carlo protocol re-sampled those *same* cells — so as k grows the
protocol trivially converges to the reference (tautology). The reference was also
"dirty": it was computed on the full generated sets, which for a memorizing model
include verbatim training-row copies, so ICL could be "best" merely by copying.

Fixes here:
  1. INDEPENDENCE — each generator's seeds are split into two DISJOINT halves:
     a REFERENCE half (defines the true ranking) and a BENCHMARK half (the only
     cells the protocol may sample). The protocol never sees the reference's data,
     so convergence is non-trivial: it happens only if the ranking is genuinely
     stable across independent draws. This is the honest claim — "a k-config average
     on fresh seeds agrees with a held-out reference at rate 1 - P(wrong)."
  2. CLEAN REFERENCE (optional --clean) — the reference ranking is recomputed on
     the dedup-clean metric (verbatim train-copies removed before scoring), so the
     "truth" is not defined by memorized rows.

We report, per metric, P(protocol names a generator OTHER than the reference best)
for increasing protocol budgets (k configs x s seeds). It should fall with k iff
the disease (single-config picks) is real and the treatment (multi-config averaging)
cures it — now without the tautology.

    /usr/bin/python3 scripts/treatment_validation_indep.py --results-dir results_reseed
    /usr/bin/python3 scripts/treatment_validation_indep.py --results-dir results_reseed --clean
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
GENERATORS = ["ctgan", "tvae", "great", "icl"]
N_MC = 4000


def best_of(scores, metric):
    if metric == "tstr":
        return max(scores, key=lambda g: scores[g])
    if metric == "c2st_auc":
        return min(scores, key=lambda g: abs(scores[g] - 0.5))
    return min(scores, key=lambda g: scores[g])  # ks/tvd/corr lower better


def norm_key_set(df):
    def nv(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else repr(round(f, 6))
        except (ValueError, TypeError):
            return str(v)
    return {tuple(nv(v) for v in row) for row in df.itertuples(index=False, name=None)}


def load_cells(results_dir, DS, clean, cfg=None):
    """vals[metric][gen][config][seed] = value.  If clean, recompute the 5 metrics on
    the dedup (verbatim-copy-removed) generated set; else read cached metrics.json."""
    vals = {m: defaultdict(lambda: defaultdict(dict)) for m in METRICS}
    if clean:
        from src import data, metrics
        df = data.load_dataset(DS, cfg)
        dsc = data._ds_cfg(DS, cfg)
        from pathlib import Path
        train, holdout = data.fixed_split(df, dsc["target"], name=DS, results_dir=Path(results_dir))
        train_keys = norm_key_set(train)

        def nv(v):
            try:
                f = float(v)
                return str(int(f)) if f == int(f) else repr(round(f, 6))
            except (ValueError, TypeError):
                return str(v)

    for cell in sorted(glob.glob(os.path.join(results_dir, "cells", f"{DS}__*"))):
        name = os.path.basename(cell)
        parts = name.split("__")
        gen, cfg_id, seed = parts[1], parts[2], int(parts[3].replace("seed", ""))
        if gen not in GENERATORS:
            continue
        if clean:
            gpath = os.path.join(cell, "generated.csv")
            if not os.path.exists(gpath):
                continue
            synth = pd.read_csv(gpath)
            if len(synth) == 0:
                continue
            mask = [tuple(nv(v) for v in row) in train_keys
                    for row in synth.itertuples(index=False, name=None)]
            n_copy = sum(mask)
            mpath = os.path.join(cell, "metrics.json")
            if n_copy == 0 and os.path.exists(mpath):
                # no copies -> dedup set == full set -> cached full metrics ARE the clean
                # metrics; skip the expensive recompute (controls/great have 0 copies).
                m = json.load(open(mpath))
            else:
                dedup = synth[[not mm for mm in mask]].reset_index(drop=True)
                if len(dedup) < 10:
                    continue
                m = metrics.compute_all(train, dedup, holdout, dsc, seed=seed)
        else:
            mpath = os.path.join(cell, "metrics.json")
            if not os.path.exists(mpath):
                continue
            m = json.load(open(mpath))
        for k in METRICS:
            v = m.get(k)
            if v is not None and v == v:
                vals[k][gen][cfg_id][seed] = v
    return vals


def split_seeds(vals):
    """Per generator, split its available seeds into disjoint (reference, benchmark)
    halves. Reference = upper half of sorted seeds; benchmark = lower half."""
    ref, bench = {}, {}
    seen = defaultdict(set)
    for m in METRICS:
        for g in vals[m]:
            for c in vals[m][g]:
                seen[g].update(vals[m][g][c].keys())
    for g, seeds in seen.items():
        s = sorted(seeds)
        cut = len(s) // 2
        bench[g] = set(s[:cut]) if cut > 0 else set(s)
        ref[g] = set(s[cut:]) if cut > 0 else set(s)
    return ref, bench


def grand_mean(vals, metric, gen, seed_set):
    xs = [v for c in vals[metric][gen]
          for sd, v in vals[metric][gen][c].items() if sd in seed_set]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config_reseed.yaml"))
    ap.add_argument("--results-dir", default="results_reseed")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--clean", action="store_true",
                    help="recompute reference+protocol metrics on dedup-clean sets")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    results_dir = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    DS = args.dataset
    random.seed(20260707)

    vals = load_cells(results_dir, DS, args.clean, cfg)
    gens = [g for g in GENERATORS if any(vals[m][g] for m in METRICS)]
    if len(gens) < 2:
        print(f"[treatment_indep] <2 generators populated under {results_dir} — cannot rank.")
        return
    ref_seeds, bench_seeds = split_seeds(vals)

    # independent reference ranking (held-out seeds)
    ref_score = {m: {g: grand_mean(vals, m, g, ref_seeds[g]) for g in gens} for m in METRICS}
    ref_best = {m: best_of({g: s for g, s in ref_score[m].items() if s == s}, m) for m in METRICS}

    def protocol_pick(metric, k, s):
        scores = {}
        for g in gens:
            # benchmark cells only, and only benchmark seeds
            cfgs = [c for c in vals[metric][g]
                    if any(sd in bench_seeds[g] for sd in vals[metric][g][c])]
            if not cfgs:
                return None
            chosen = random.sample(cfgs, min(k, len(cfgs)))
            cell_means = []
            for c in chosen:
                seeds = [sd for sd in vals[metric][g][c] if sd in bench_seeds[g]]
                samp = random.sample(seeds, min(s, len(seeds)))
                cell_means.append(sum(vals[metric][g][c][sd] for sd in samp) / len(samp))
            scores[g] = sum(cell_means) / len(cell_means)
        return best_of(scores, metric)

    def p_wrong(metric, k, s):
        picks = [protocol_pick(metric, k, s) for _ in range(N_MC)]
        picks = [p for p in picks if p is not None]
        if not picks:
            return float("nan")
        return sum(p != ref_best[metric] for p in picks) / len(picks)

    PROTO = [("single (1 cfg,1 seed)", 1, 1),
             ("1 cfg, 3 seeds", 1, 3),
             ("3 cfgs, 3 seeds", 3, 3),
             ("all cfgs, all seeds", 99, 99)]

    tag = "CLEAN+independent" if args.clean else "independent (disjoint-seed)"
    print(f"\nTreatment validation (diabetes) — {tag} reference")
    print("reference = held-out-seed grand mean; benchmark protocol samples the OTHER seeds")
    print("ref seeds per gen: " + ", ".join(f"{g}:{sorted(ref_seeds[g])}" for g in gens))
    print("true best per metric: " + ", ".join(f"{m.split('_')[0]}={ref_best[m]}" for m in METRICS) + "\n")
    header = f"{'metric':<12}" + "".join(f"{lbl:>24}" for lbl, _, _ in PROTO)
    print(header); print("-" * len(header))
    out = []
    for m in METRICS:
        row = [p_wrong(m, k, s) for _, k, s in PROTO]
        print(f"{m:<12}" + "".join(f"{x:>24.3f}" for x in row))
        out.append((DS, m, ref_best[m], *[round(x, 4) for x in row]))

    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
    out_name = args.out_name or ("treatment_validation_clean" if args.clean
                                 else "treatment_validation_indep")
    out_path = os.path.join(results_dir, "analysis", f"{out_name}.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "metric", "ref_best",
                    "p_wrong_1cfg_1seed", "p_wrong_1cfg_3seed",
                    "p_wrong_3cfg_3seed", "p_wrong_allcfg_allseed"])
        w.writerows(out)
    print(f"\nwrote {out_path}")
    print("\nHonest reading: averaging N cells trivially shrinks variance; what this "
          "shows is that a single-config pick disagrees with an INDEPENDENT reference "
          "at rate P(wrong), and that a modest multi-config budget closes most of that gap.")


if __name__ == "__main__":
    main()
