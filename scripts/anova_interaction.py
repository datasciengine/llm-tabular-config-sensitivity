#!/usr/bin/env python3
"""re-specify the RQ3 ANOVA with a generator x config INTERACTION term.

The problem: reporting `config` as a main effect (33% of KS variance) is
misleading, because only ICL responds to configuration — GReaT does not. The apparent
"config main effect" is largely a generator x config INTERACTION, which the main-effects-
only model dumps into the residual. Adding the interaction should (a) shrink the config
main effect, (b) make the interaction the dominant configuration-related term, and
(c) cut the residual — directly showing configuration sensitivity is ICL-specific.

We fit, per metric, on the LLM generators (great, icl):
    metric ~ C(generator) + C(config) + C(generator):C(config) + C(seed)
Type-II ANOVA eta^2 (% of variance). Two views:
  (A) SHARED configs only (present for BOTH generators -> balanced, cleanly estimable
      interaction): baseline, column_order perm1/2/3, numeric_format=rounded.
  (B) ALL configs (adds ICL-only serialization=keyvalue/compact; unbalanced).

    /usr/bin/python3 scripts/anova_interaction.py --results-dir results_reseed
"""
import argparse
import glob
import json
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
LLM = ["great", "icl"]


def load(results_dir, ds):
    rows = []
    for c in sorted(glob.glob(os.path.join(results_dir, "cells", f"{ds}__*"))):
        parts = os.path.basename(c).split("__")
        gen, cfg, seed = parts[1], parts[2], int(parts[3].replace("seed", ""))
        if gen not in LLM:
            continue
        mp = os.path.join(c, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        rows.append({"generator": gen, "config_id": cfg, "seed": seed,
                     **{k: m.get(k) for k in METRICS}})
    return pd.DataFrame(rows)


def eta_sq(df, metric, interaction=True):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    sub = df[["generator", "config_id", "seed", metric]].dropna()
    terms = ["C(generator)", "C(config_id)", "C(seed)"]
    if interaction:
        terms.insert(2, "C(generator):C(config_id)")
    # keep only factors with >1 level
    keep = []
    for t in terms:
        if t == "C(generator):C(config_id)":
            keep.append(t)
        else:
            f = t[2:-1]
            if sub[f].nunique() > 1:
                keep.append(t)
    formula = f"Q('{metric}') ~ " + " + ".join(keep)
    model = smf.ols(formula, data=sub).fit()
    aov = anova_lm(model, typ=2)
    ss_tot = aov["sum_sq"].sum()
    out = {}
    label = {"C(generator)": "gen", "C(config_id)": "config",
             "C(generator):C(config_id)": "gen×config", "C(seed)": "seed",
             "Residual": "resid"}
    for term, lab in label.items():
        out[lab] = 100 * aov.loc[term, "sum_sq"] / ss_tot if term in aov.index and ss_tot > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_reseed")
    ap.add_argument("--dataset", default="diabetes")
    args = ap.parse_args()
    results_dir = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    df = load(results_dir, args.dataset)

    shared = sorted(set.intersection(*[set(df[df.generator == g]["config_id"]) for g in LLM]))
    print(f"generators={LLM}  shared configs ({len(shared)}): {shared}")
    print(f"ICL-only: {sorted(set(df[df.generator=='icl']['config_id']) - set(shared))}\n")

    import csv as _csv
    csv_rows = []
    for view, title, sub in [("shared", "(A) SHARED configs — balanced, clean interaction",
                              df[df.config_id.isin(shared)]),
                             ("all", "(B) ALL configs — unbalanced (ICL-only serialization added)",
                              df)]:
        print(f"=== {title} ===")
        print(f"{'metric':<16}{'gen':>7}{'config':>8}{'gen×config':>12}{'seed':>7}{'resid':>8}")
        for m in METRICS:
            try:
                e = eta_sq(sub, m, interaction=True)
                print(f"{m:<16}{e['gen']:>7.1f}{e['config']:>8.1f}{e['gen×config']:>12.1f}"
                      f"{e['seed']:>7.1f}{e['resid']:>8.1f}")
                csv_rows.append((view, m, round(e['gen'], 2), round(e['config'], 2),
                                 round(e['gen×config'], 2), round(e['seed'], 2), round(e['resid'], 2)))
            except Exception as ex:
                print(f"{m:<16} FAILED: {type(ex).__name__}: {ex}")
        print()

    out = os.path.join(results_dir, "analysis", "variance_interaction.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["view", "metric", "pct_gen", "pct_config", "pct_gen_x_config", "pct_seed", "pct_resid"])
        w.writerows(csv_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
