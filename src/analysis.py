"""Analysis & deliverables. Run only AFTER results/ is populated.

Produces (the project spec sec 11 + config.yaml `analysis` block):
  1. per-metric tables: mean +/- 95% CI, per dataset x generator x config
  2. variance-components: % metric variance from generator/dataset/config/seed/residual
     (LLM generators only; controls reported separately as the seed-only floor)
  3. ranking-stability: Kendall's W across configs, rank-flip rate, top-1 flip freq,
     P(wrong conclusion) under a single-config/single-seed pick
  4. CSI: normalized spread of the primary fidelity metric across configs, per generator
  5. Holm-corrected significance: two families (config-effect, ranking)
  6. figures: per-config spread, rank-flip, variance breakdown, CSI bar

Partial-friendly: analyzes whatever cells exist and prints coverage; never assumes
all 240 cells are present. Run: `python -m src.analysis [--results-dir results]`.

Metric directions (orientation -> "goodness", higher = better, for ranking):
  ks_marginal, tvd_categorical, corr_diff : lower better  -> negate
  c2st_auc                                 : 0.5 best      -> -|auc - 0.5|
  tstr                                     : higher better -> as-is
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]
LOWER_BETTER = {"ks_marginal", "tvd_categorical", "corr_diff"}
LLM_GENERATORS = ["great", "icl"]
CONTROL_GENERATORS = ["ctgan", "tvae"]
PRIMARY_METRIC = "ks_marginal"
BASELINE_CONFIG = "baseline"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_results(results_dir: str | Path = "results") -> pd.DataFrame:
    """Read every cell's metrics.json + gen_meta.json into one tidy frame.
    One row per successful cell; columns: dataset, generator, config_id, seed,
    is_llm, icl_yield, <5 metrics>."""
    cells = Path(results_dir) / "cells"
    rows = []
    for d in sorted(cells.glob("*/")):
        mt_p, gm_p = d / "metrics.json", d / "gen_meta.json"
        gm = json.loads(gm_p.read_text()) if gm_p.exists() else {}
        if gm.get("status") not in (None, "ok"):       # skip errored/crashed cells
            continue
        if not mt_p.exists():
            continue
        mt = json.loads(mt_p.read_text())
        rows.append({
            "dataset": mt.get("dataset"),
            "generator": mt.get("generator"),
            "config_id": mt.get("config_id"),
            "seed": mt.get("seed"),
            "is_llm": mt.get("generator") in LLM_GENERATORS,
            "icl_yield": gm.get("icl_valid_yield"),
            **{m: mt.get(m) for m in METRICS},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No completed cells found under {cells}. Run the grid first.")
    return df


def _goodness(metric: str, vals: pd.Series | np.ndarray) -> np.ndarray:
    """Orient a metric so higher = better (for ranking/flip analysis)."""
    v = np.asarray(vals, dtype=float)
    if metric in LOWER_BETTER:
        return -v
    if metric == "c2st_auc":
        return -np.abs(v - 0.5)
    return v                                            # tstr: higher better


# ----------------------------------------------------------------------------
# 1. Per-metric tables: mean +/- 95% CI (bootstrap, honoring config bootstrap_ci)
# ----------------------------------------------------------------------------
def _bootstrap_ci(vals, level=0.95, n_boot=1000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    if vals.size == 1:
        return (float(vals[0]), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return (float(vals.mean()), float(lo), float(hi))


def per_metric_tables(df: pd.DataFrame, level=0.95, n_boot=1000) -> pd.DataFrame:
    """Long table: one row per (dataset, generator, config_id, metric) with
    mean, ci_lo, ci_hi (bootstrap across seeds) and n_seeds."""
    out = []
    keys = ["dataset", "generator", "config_id"]
    for (ds, gen, cid), g in df.groupby(keys):
        for m in METRICS:
            mean, lo, hi = _bootstrap_ci(g[m].values, level, n_boot)
            out.append({"dataset": ds, "generator": gen, "config_id": cid,
                        "metric": m, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                        "n_seeds": int(g[m].notna().sum())})
    return pd.DataFrame(out).sort_values(["metric", "dataset", "generator", "config_id"])


# ----------------------------------------------------------------------------
# 2. Variance components (LLM only) + control seed-only floor
# ----------------------------------------------------------------------------
def variance_components(df: pd.DataFrame) -> pd.DataFrame:
    """% of each metric's variance attributable to generator/dataset/config/seed
    /residual, within the LLM generators. Primary method: ANOVA eta-squared from an
    OLS fit with all four factors as categoricals (robust, always available, satisfies
    the project spec sec 6's 'statsmodels ANOVA'); a MixedLM cross-check on config+seed is
    attempted and reported when it converges. The headline (RQ3) is config vs seed."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    llm = df[df["is_llm"]].copy()
    factors = ["generator", "dataset", "config_id", "seed"]
    out = []
    for m in METRICS:
        sub = llm[["dataset", "generator", "config_id", "seed", m]].dropna()
        if sub[m].nunique() < 2 or len(sub) < 8:
            continue
        # Drop factors with a single level (would alias / break the design).
        use = [f for f in factors if sub[f].nunique() > 1]
        formula = f"Q('{m}') ~ " + " + ".join(f"C({f})" for f in use)
        try:
            model = smf.ols(formula, data=sub).fit()
            aov = anova_lm(model, typ=2)
            ss_total = aov["sum_sq"].sum()
            row = {"metric": m, "method": "anova_eta_sq", "n": len(sub)}
            for f in factors:
                term = f"C({f})"
                row[f"pct_{f}"] = (100 * aov.loc[term, "sum_sq"] / ss_total
                                   if term in aov.index and ss_total > 0 else np.nan)
            row["pct_residual"] = (100 * aov.loc["Residual", "sum_sq"] / ss_total
                                   if "Residual" in aov.index and ss_total > 0 else np.nan)
            out.append(row)
        except Exception as e:                          # pragma: no cover
            out.append({"metric": m, "method": f"FAILED: {type(e).__name__}", "n": len(sub)})

        # MixedLM cross-check: config + seed as variance components (dataset:gen groups).
        # This is a secondary estimate; it can fail to converge on tiny/near-degenerate
        # variance, so its noisy warnings are suppressed and failures are ignored.
        try:
            import warnings
            sub2 = sub.copy()
            sub2["grp"] = sub2["dataset"].astype(str) + ":" + sub2["generator"].astype(str)
            md = smf.mixedlm(f"Q('{m}') ~ 1", sub2, groups="grp",
                             vc_formula={"config": "0 + C(config_id)", "seed": "0 + C(seed)"})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mf = md.fit(reml=True, method="lbfgs", maxiter=200)
            vc = mf.vcomp
            resid = mf.scale
            tot = float(np.nansum(vc)) + resid
            if tot > 0 and len(vc) >= 2:
                out.append({"metric": m, "method": "mixedlm",
                            "pct_config": 100 * vc[0] / tot, "pct_seed": 100 * vc[1] / tot,
                            "pct_residual": 100 * resid / tot, "n": len(sub)})
        except Exception:
            pass                                        # mixed model is a cross-check only
    return pd.DataFrame(out)


def control_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-only floor: across-seed std for each control generator (baseline only),
    per (dataset, metric). The headline test (the project spec sec 12.4) compares LLM
    across-config std against this floor."""
    ctrl = df[df["generator"].isin(CONTROL_GENERATORS)]
    out = []
    for (ds, gen), g in ctrl.groupby(["dataset", "generator"]):
        for m in METRICS:
            out.append({"dataset": ds, "generator": gen, "metric": m,
                        "seed_std": float(g[m].std(ddof=1)), "n_seeds": int(g[m].notna().sum())})
    return pd.DataFrame(out)


def config_vs_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Headline (RQ1/RQ3): for each LLM generator, across-config std (of seed-mean per
    config) vs the matching-dataset control seed-only floor (mean over controls)."""
    floor = control_floor(df)
    floor_by = floor.groupby(["dataset", "metric"])["seed_std"].mean()
    out = []
    llm = df[df["is_llm"]]
    for (ds, gen), g in llm.groupby(["dataset", "generator"]):
        for m in METRICS:
            config_means = g.groupby("config_id")[m].mean()
            cfg_std = float(config_means.std(ddof=1)) if config_means.notna().sum() > 1 else np.nan
            fl = float(floor_by.get((ds, m), np.nan))
            out.append({"dataset": ds, "generator": gen, "metric": m,
                        "llm_config_std": cfg_std, "control_seed_floor": fl,
                        "ratio_config_over_floor": cfg_std / fl if fl and np.isfinite(fl) and fl > 0 else np.nan})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# 3. Ranking stability
# ----------------------------------------------------------------------------
def _kendalls_w(rank_matrix: np.ndarray) -> float:
    """rank_matrix: rows = judges (configs), cols = objects (generators), entries =
    ranks (1=best). Returns Kendall's W in [0,1]; 1 = perfect agreement across configs."""
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return np.nan
    Rj = rank_matrix.sum(axis=0)
    S = float(np.sum((Rj - Rj.mean()) ** 2))
    denom = m ** 2 * (n ** 3 - n)
    return 12 * S / denom if denom > 0 else np.nan


def _config_ranking(df: pd.DataFrame, ds: str, metric: str):
    """For one (dataset, metric): build a configs x generators matrix of ranks (1=best),
    using each generator's seed-mean goodness within each config. Controls (baseline
    only) are broadcast to every config so all generators are comparable per config."""
    sub = df[df["dataset"] == ds]
    gens = sorted(sub["generator"].unique())
    configs = sorted(sub[sub["is_llm"]]["config_id"].unique()) or [BASELINE_CONFIG]
    mat, kept_cfgs = [], []
    for cid in configs:
        goods = []
        for gen in gens:
            if gen in CONTROL_GENERATORS:
                cell = sub[(sub["generator"] == gen) & (sub["config_id"] == BASELINE_CONFIG)]
            else:
                cell = sub[(sub["generator"] == gen) & (sub["config_id"] == cid)]
            goods.append(np.nanmean(_goodness(metric, cell[metric].values)) if len(cell) else np.nan)
        if np.isfinite(goods).sum() >= 2:
            from scipy.stats import rankdata
            g = np.array(goods, dtype=float)
            ranks = rankdata(-np.where(np.isfinite(g), g, -np.inf), method="average")
            mat.append(ranks)
            kept_cfgs.append(cid)
    return gens, kept_cfgs, np.array(mat)


def ranking_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, metric): Kendall's W across configs, rank-flip rate (fraction of
    config pairs whose generator ordering differs), top-1 flip frequency (fraction of
    configs whose best generator differs from the baseline config's), and P(wrong
    conclusion) = fraction of single-(config,seed) picks whose top-1 generator differs
    from the all-configs/all-seeds 'true' best."""
    out = []
    for ds in sorted(df["dataset"].unique()):
        for m in METRICS:
            gens, cfgs, mat = _config_ranking(df, ds, m)
            if mat.size == 0 or len(cfgs) < 1:
                continue
            W = _kendalls_w(mat) if len(cfgs) >= 2 else np.nan
            # rank-flip rate: order disagreement over all config pairs
            flips, pairs = 0, 0
            for i, j in itertools.combinations(range(len(cfgs)), 2):
                pairs += 1
                if not np.array_equal(np.argsort(mat[i]), np.argsort(mat[j])):
                    flips += 1
            flip_rate = flips / pairs if pairs else np.nan
            # top-1 flip frequency vs baseline config's top-1
            top1 = [gens[int(np.argmin(r))] for r in mat]   # rank 1 = best = argmin
            base_top1 = top1[cfgs.index(BASELINE_CONFIG)] if BASELINE_CONFIG in cfgs else top1[0]
            top1_flip = float(np.mean([t != base_top1 for t in top1]))
            p_wrong = _p_wrong_single_run(df, ds, m, gens)
            out.append({"dataset": ds, "metric": m, "n_configs": len(cfgs),
                        "kendall_w": W, "rank_flip_rate": flip_rate,
                        "top1_flip_freq": top1_flip, "p_wrong_single_run": p_wrong,
                        "true_best": _true_best(df, ds, m, gens)})
    return pd.DataFrame(out)


def _true_best(df, ds, metric, gens):
    """Generator with the best mean goodness over ALL configs+seeds (the 'truth')."""
    sub = df[df["dataset"] == ds]
    means = {}
    for g in gens:
        vals = sub[sub["generator"] == g][metric].values
        means[g] = float(np.nanmean(_goodness(metric, vals))) if len(vals) else -np.inf
    return max(means, key=means.get)


def _p_wrong_single_run(df, ds, metric, gens) -> float:
    """Enumerate every (config, seed) pick; for each, rank generators by their single
    observed value and check whether the top-1 matches the 'true' best (all data)."""
    sub = df[df["dataset"] == ds]
    true_best = _true_best(df, ds, metric, gens)
    configs = sorted(sub[sub["is_llm"]]["config_id"].unique()) or [BASELINE_CONFIG]
    seeds = sorted(sub["seed"].dropna().unique())
    wrong = total = 0
    for cid in configs:
        for s in seeds:
            goods = {}
            for gen in gens:
                cfg = BASELINE_CONFIG if gen in CONTROL_GENERATORS else cid
                cell = sub[(sub["generator"] == gen) & (sub["config_id"] == cfg) & (sub["seed"] == s)]
                if len(cell):
                    goods[gen] = float(np.nanmean(_goodness(metric, cell[metric].values)))
            if len(goods) < 2:
                continue
            total += 1
            if max(goods, key=goods.get) != true_best:
                wrong += 1
    return wrong / total if total else np.nan


# ----------------------------------------------------------------------------
# 4. CSI — Configuration Sensitivity Index
# ----------------------------------------------------------------------------
def configuration_sensitivity_index(df: pd.DataFrame, primary_metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """CSI per (generator, dataset) = normalized spread of the primary fidelity metric
    across task-irrelevant configs = std(config means) / |mean(config means)|
    (coefficient of variation). A simple diagnostic, NOT a new metric (the project spec sec 6).
    Controls have one config -> CSI ~ 0 by construction (sanity check)."""
    out = []
    for (ds, gen), g in df.groupby(["dataset", "generator"]):
        config_means = g.groupby("config_id")[primary_metric].mean()
        if config_means.notna().sum() < 2:
            csi = 0.0 if gen in CONTROL_GENERATORS else np.nan
        else:
            mu = float(np.abs(config_means.mean()))
            csi = float(config_means.std(ddof=1) / mu) if mu > 0 else np.nan
        out.append({"dataset": ds, "generator": gen, "metric": primary_metric,
                    "csi": csi, "n_configs": int(config_means.notna().sum())})
    csi_df = pd.DataFrame(out)
    # also an aggregate CSI per generator (mean over datasets)
    agg = (csi_df[csi_df["csi"].notna()].groupby("generator")["csi"].mean()
           .rename("csi_mean_over_datasets").reset_index())
    return csi_df.merge(agg, on="generator", how="left")


# ----------------------------------------------------------------------------
# 5. Holm-corrected significance (two families; the project spec sec 12.8)
# ----------------------------------------------------------------------------
def _holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN-safe; preserves order)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    adj = [np.nan] * len(pvals)
    if not idx:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def holm_config_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Family = (llm_generator, dataset, metric); members = each non-baseline config.
    Paired t-test (across shared seeds) of config-vs-baseline; Holm within each family."""
    from scipy.stats import ttest_rel
    rows = []
    llm = df[df["is_llm"]]
    for (gen, ds), g in llm.groupby(["generator", "dataset"]):
        for m in METRICS:
            base = g[g["config_id"] == BASELINE_CONFIG].set_index("seed")[m]
            fam = []
            for cid in sorted(g["config_id"].unique()):
                if cid == BASELINE_CONFIG:
                    continue
                cur = g[g["config_id"] == cid].set_index("seed")[m]
                common = base.index.intersection(cur.index)
                if len(common) >= 2 and np.isfinite(base[common]).all() and np.isfinite(cur[common]).all():
                    try:
                        t, p = ttest_rel(cur[common], base[common])
                    except Exception:
                        t, p = np.nan, np.nan
                    fam.append({"generator": gen, "dataset": ds, "metric": m, "config_id": cid,
                                "delta_mean": float(cur[common].mean() - base[common].mean()),
                                "p_raw": float(p) if np.isfinite(p) else np.nan})
            for r, padj in zip(fam, _holm([x["p_raw"] for x in fam])):
                r["p_holm"] = padj
                r["sig_0.05"] = bool(padj is not None and np.isfinite(padj) and padj < 0.05)
                rows.append(r)
    return pd.DataFrame(rows)


def holm_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Family = (dataset, metric); members = generator pairs. Paired t-test across
    seeds at the baseline config (the canonical single-config comparison RQ2 questions);
    Holm within each family."""
    from scipy.stats import ttest_rel
    rows = []
    base = df[df["config_id"] == BASELINE_CONFIG]
    for ds in sorted(base["dataset"].unique()):
        sub = base[base["dataset"] == ds]
        gens = sorted(sub["generator"].unique())
        for m in METRICS:
            fam = []
            for a, b in itertools.combinations(gens, 2):
                va = sub[sub["generator"] == a].set_index("seed")[m]
                vb = sub[sub["generator"] == b].set_index("seed")[m]
                common = va.index.intersection(vb.index)
                if len(common) >= 2 and np.isfinite(va[common]).all() and np.isfinite(vb[common]).all():
                    try:
                        t, p = ttest_rel(va[common], vb[common])
                    except Exception:
                        t, p = np.nan, np.nan
                    fam.append({"dataset": ds, "metric": m, "pair": f"{a} vs {b}",
                                "delta_mean": float(va[common].mean() - vb[common].mean()),
                                "p_raw": float(p) if np.isfinite(p) else np.nan})
            for r, padj in zip(fam, _holm([x["p_raw"] for x in fam])):
                r["p_holm"] = padj
                r["sig_0.05"] = bool(padj is not None and np.isfinite(padj) and padj < 0.05)
                rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, out_dir: str | Path,
                 var_df: pd.DataFrame | None = None,
                 rank_df: pd.DataFrame | None = None,
                 csi_df: pd.DataFrame | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Fig 1: per-config spread of the primary metric (LLM gens), one panel per dataset
    llm = df[df["is_llm"]]
    datasets = sorted(llm["dataset"].unique())
    if datasets:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            sub = llm[llm["dataset"] == ds]
            for gen in sorted(sub["generator"].unique()):
                gg = sub[sub["generator"] == gen]
                cfgs = sorted(gg["config_id"].unique())
                means = [gg[gg["config_id"] == c][PRIMARY_METRIC].mean() for c in cfgs]
                ax.plot(range(len(cfgs)), means, marker="o", label=gen)
                ax.set_xticks(range(len(cfgs)))
                ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{ds}: {PRIMARY_METRIC} across configs")
            ax.set_ylabel(PRIMARY_METRIC)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / "fig1_per_config_spread.png"
        fig.savefig(p, dpi=130); plt.close(fig); saved.append(str(p))

    # Fig 2: rank-flip rate per dataset (primary metric)
    if rank_df is not None and not rank_df.empty:
        rp = rank_df[rank_df["metric"] == PRIMARY_METRIC]
        if not rp.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(rp["dataset"], rp["rank_flip_rate"])
            ax.set_ylabel("rank-flip rate"); ax.set_ylim(0, 1)
            ax.set_title(f"Rank-flip rate across configs ({PRIMARY_METRIC})")
            fig.tight_layout()
            p = out_dir / "fig2_rank_flip.png"
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(str(p))

    # Fig 3: variance breakdown (stacked bar per metric, ANOVA rows)
    if var_df is not None and not var_df.empty:
        vv = var_df[var_df["method"] == "anova_eta_sq"]
        comp = ["pct_generator", "pct_dataset", "pct_config_id", "pct_seed", "pct_residual"]
        comp = [c for c in comp if c in vv.columns]
        if not vv.empty and comp:
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(vv))
            bottom = np.zeros(len(vv))
            for c in comp:
                vals = vv[c].fillna(0).values
                ax.bar(x, vals, bottom=bottom, label=c.replace("pct_", ""))
                bottom += vals
            ax.set_ylabel("% variance"); ax.set_title("Variance components (LLM generators)")
            ax.legend(fontsize=8, ncol=3)
            ax.set_xticks(x); ax.set_xticklabels(vv["metric"], rotation=30, ha="right")
            fig.tight_layout()
            p = out_dir / "fig3_variance_breakdown.png"
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(str(p))

    # Fig 4: CSI bar per (generator, dataset)
    if csi_df is not None and not csi_df.empty:
        cc = csi_df.dropna(subset=["csi"])
        if not cc.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            pivot = cc.pivot_table(index="generator", columns="dataset", values="csi")
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel(f"CSI ({PRIMARY_METRIC}, CV across configs)")
            ax.set_title("Configuration Sensitivity Index")
            fig.tight_layout()
            p = out_dir / "fig4_csi.png"
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(str(p))

    return saved


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_analysis(results_dir: str | Path = "results", out_dir: str | Path | None = None) -> dict:
    df = load_results(results_dir)
    out_dir = Path(out_dir) if out_dir else Path(results_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # coverage report
    cov = (df.groupby(["dataset", "generator"])["config_id"].nunique()
           .rename("n_configs").reset_index())
    n_seeds = df.groupby(["dataset", "generator", "config_id"]).size().rename("n_seeds")
    print(f"[analysis] loaded {len(df)} cells; "
          f"{df['dataset'].nunique()} datasets, {df['generator'].nunique()} generators")
    print(cov.to_string(index=False))

    tables = {
        "per_metric_ci": per_metric_tables(df),
        "variance_components": variance_components(df),
        "control_floor": control_floor(df),
        "config_vs_floor": config_vs_floor(df),
        "ranking_stability": ranking_stability(df),
        "csi": configuration_sensitivity_index(df),
        "holm_config_effects": holm_config_effects(df),
        "holm_ranking": holm_ranking(df),
        "coverage": cov,
    }
    for name, t in tables.items():
        path = out_dir / f"{name}.csv"
        t.to_csv(path, index=False)
        print(f"[analysis] wrote {path}  ({len(t)} rows)")

    figs = make_figures(df, out_dir / "figures",
                        var_df=tables["variance_components"],
                        rank_df=tables["ranking_stability"],
                        csi_df=tables["csi"])
    print(f"[analysis] wrote {len(figs)} figures to {out_dir / 'figures'}")
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Variance / ranking / CSI analysis over results/cells.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    run_analysis(args.results_dir, args.out_dir)
