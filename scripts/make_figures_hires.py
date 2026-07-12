#!/usr/bin/env python3
"""publication-quality, high-resolution figures.

Earlier figures were 130-DPI PNGs with small fonts and poor readability.
This regenerates the four paper figures as BOTH 600-DPI PNG and vector PDF, with a
colorblind-safe palette, larger fonts, and grids. It is pure plotting: it reads the
CSV tables that `src/analysis.py` already wrote (per_metric_ci, ranking_stability,
variance_components, csi), so it is fast and never re-runs the heavy stats.

Run AFTER analysis.py has populated <results-dir>/analysis/:
    /usr/bin/python3 scripts/make_figures_hires.py --results-dir results_reseed
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY = "ks_marginal"
# Okabe-Ito colorblind-safe palette
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "figure.autolayout": False,
})

PRETTY_CFG = {
    "baseline": "baseline",
    "serialization=keyvalue": "ser:keyvalue",
    "serialization=compact": "ser:compact",
    "column_order=perm1": "ord:perm1",
    "column_order=perm2": "ord:perm2",
    "column_order=perm3": "ord:perm3",
    "numeric_format=rounded": "num:rounded",
}


def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return os.path.join(out_dir, f"{name}.pdf")


def fig1_per_config_spread(per_metric, out_dir):
    """Per-config spread of the primary metric, one panel per dataset, LLM gens."""
    df = per_metric[(per_metric["metric"] == PRIMARY)]
    datasets = sorted(df["dataset"].unique())
    if not datasets:
        return None
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["dataset"] == ds]
        gens = [g for g in ["great", "icl"] if g in set(sub["generator"])]
        for gi, gen in enumerate(gens):
            gg = sub[sub["generator"] == gen].copy()
            cfgs = [c for c in PRETTY_CFG if c in set(gg["config_id"])]
            means = [gg[gg["config_id"] == c]["mean"].mean() for c in cfgs]
            los = [gg[gg["config_id"] == c]["ci_lo"].mean() for c in cfgs]
            his = [gg[gg["config_id"] == c]["ci_hi"].mean() for c in cfgs]
            yerr = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
            ax.errorbar(range(len(cfgs)), means, yerr=yerr, marker="o", capsize=3,
                        color=CB[gi], label=gen, linewidth=1.8, markersize=6)
            ax.set_xticks(range(len(cfgs)))
            ax.set_xticklabels([PRETTY_CFG[c] for c in cfgs], rotation=40, ha="right")
        ax.set_title(f"{ds}")
        ax.set_ylabel(f"{PRIMARY} (mean ± 95% CI)")
        ax.legend(title="generator", frameon=False)
    fig.suptitle("Primary fidelity metric across task-irrelevant configurations", y=1.02)
    return _save(fig, out_dir, "fig1_per_config_spread")


def fig2_rank_flip(rank_df, out_dir):
    # Per-metric rank-flip for the primary (single) dataset, with Kendall's W labels.
    if rank_df.empty:
        return None
    order = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
    pretty = {"ks_marginal": "KS", "tvd_categorical": "TVD", "corr_diff": "corr",
              "c2st_auc": "C2ST", "tstr": "TSTR"}
    rp = rank_df.set_index("metric").reindex(order).dropna(subset=["rank_flip_rate"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(rp)), rp["rank_flip_rate"], color=CB[0], width=0.62)
    for i, (m, r) in enumerate(rp.iterrows()):
        w = r.get("kendall_w")
        if pd.notna(w):
            ax.text(i, r["rank_flip_rate"] + 0.02, f"$W$={w:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(rp)))
    ax.set_xticklabels([pretty.get(m, m) for m in rp.index])
    ax.set_ylabel("rank-flip rate across configs")
    ax.set_ylim(0, 1.08)
    ax.set_title("Ranking instability by metric (diabetes, 15 seeds)")
    return _save(fig, out_dir, "fig2_rank_flip")


def fig3_variance(var_df, out_dir):
    # var_df is the INTERACTION decomposition (variance_interaction.csv, view=shared):
    # columns pct_gen, pct_config, pct_gen_x_config, pct_seed, pct_resid.
    vv = var_df[var_df.get("view", "shared") == "shared"].copy() if "view" in var_df.columns else var_df.copy()
    comp = [c for c in ["pct_gen", "pct_config", "pct_gen_x_config", "pct_seed", "pct_resid"]
            if c in vv.columns]
    if vv.empty or not comp:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(vv))
    bottom = np.zeros(len(vv))
    labels = {"pct_gen": "generator", "pct_config": "config (main)",
              "pct_gen_x_config": "gen × config", "pct_seed": "seed", "pct_resid": "residual"}
    for ci, c in enumerate(comp):
        vals = vv[c].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=labels[c], color=CB[ci])
        bottom += vals
    ax.set_ylabel("% of variance")
    ax.set_title("Variance components with generator × configuration interaction\n"
                 "(diabetes, 15 seeds; shared configs)")
    pretty = {"ks_marginal": "KS", "tvd_categorical": "TVD", "corr_diff": "corr",
              "c2st_auc": "C2ST", "tstr": "TSTR"}
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(m, m) for m in vv["metric"]], rotation=0)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    return _save(fig, out_dir, "fig3_variance_breakdown")


def fig4_csi(csi_df, out_dir):
    cc = csi_df.dropna(subset=["csi"])
    if cc.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pivot = cc.pivot_table(index="generator", columns="dataset", values="csi")
    pivot.plot(kind="bar", ax=ax, color=CB[:pivot.shape[1]], width=0.7)
    ax.set_ylabel(f"CSI (CV of {PRIMARY} across configs)")
    ax.set_title("Configuration Sensitivity Index")
    ax.legend(title="dataset", frameon=False)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save(fig, out_dir, "fig4_csi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_reseed")
    args = ap.parse_args()
    adir = os.path.join(ROOT, args.results_dir, "analysis") \
        if not os.path.isabs(args.results_dir) else os.path.join(args.results_dir, "analysis")
    out_dir = os.path.join(adir, "figures_hires")
    os.makedirs(out_dir, exist_ok=True)

    def load(name):
        p = os.path.join(adir, f"{name}.csv")
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    per_metric = load("per_metric_ci")
    rank_df = load("ranking_stability")
    var_df = load("variance_interaction")   # interaction decomposition
    csi_df = load("csi")

    made = []
    if not per_metric.empty:
        made.append(fig1_per_config_spread(per_metric, out_dir))
    if not rank_df.empty:
        made.append(fig2_rank_flip(rank_df, out_dir))
    if not var_df.empty:
        made.append(fig3_variance(var_df, out_dir))
    if not csi_df.empty:
        made.append(fig4_csi(csi_df, out_dir))
    made = [m for m in made if m]
    print(f"[make_figures_hires] wrote {len(made)} figures (PDF+600dpi PNG) to {out_dir}")
    for m in made:
        print(f"  {m}")
    if not made:
        print("  (no analysis CSVs found — run src/analysis.py on this results-dir first)")


if __name__ == "__main__":
    main()
