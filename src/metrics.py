"""The 5 FROZEN metrics. No new metric may be added (see the project spec).

All take (real_df, synth_df, ds_cfg) and return a float. Metrics are ALWAYS computed
on canonical column order (the runner realigns synth to real's columns first); the
config's column-order perturbation reaches ONLY the LLM during fit, never the metrics.

Sanity rule: real-vs-real should score ~perfect (KS~0, TVD~0, corr_diff~0, C2ST~0.5,
TSTR~the real baseline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALL_METRICS = ["ks_marginal", "tvd_categorical", "corr_diff", "c2st_auc", "tstr"]


# ----------------------------------------------------------------------------
# Column typing (kept local so metrics don't depend on a live config object)
# ----------------------------------------------------------------------------
def _types(real: pd.DataFrame, ds_cfg: dict):
    """(numeric, categorical, target). Categorical = object/category/bool + target."""
    target = ds_cfg.get("target")
    numeric, categorical = [], []
    for c in real.columns:
        # Numeric iff genuine numeric dtype (is_numeric_dtype is robust to pandas reading
        # strings as the newer 'string' dtype vs 'object' — else categoricals get typed as
        # numeric and TVD/one-hot/KS are computed on the wrong columns).
        if c == target or not pd.api.types.is_numeric_dtype(real[c]):
            categorical.append(c)
        else:
            numeric.append(c)
    return numeric, categorical, target


def _build_preprocessor(numeric, categorical):
    """One-hot categoricals (ignore unseen) + standardize numerics. Shared by C2ST
    and TSTR so the two synth/real pipelines are identical."""
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop")


# ----------------------------------------------------------------------------
# 1. KS over numeric marginals
# ----------------------------------------------------------------------------
def ks_marginal(real, synth, ds_cfg) -> float:           # lower better
    numeric, _, _ = _types(real, ds_cfg)
    if not numeric:
        return float("nan")
    stats = []
    for c in numeric:
        r = pd.to_numeric(real[c], errors="coerce").dropna()
        s = pd.to_numeric(synth[c], errors="coerce").dropna()
        if len(r) and len(s):
            stats.append(ks_2samp(r, s).statistic)
    return float(np.mean(stats)) if stats else float("nan")


# ----------------------------------------------------------------------------
# 2. TVD over categorical marginals
# ----------------------------------------------------------------------------
def tvd_categorical(real, synth, ds_cfg) -> float:        # lower better
    _, categorical, _ = _types(real, ds_cfg)
    if not categorical:
        return float("nan")
    tvds = []
    for c in categorical:
        rp = real[c].astype(str).value_counts(normalize=True)
        sp = synth[c].astype(str).value_counts(normalize=True)
        cats = rp.index.union(sp.index)
        rp = rp.reindex(cats, fill_value=0.0)
        sp = sp.reindex(cats, fill_value=0.0)
        tvds.append(0.5 * float(np.abs(rp.values - sp.values).sum()))
    return float(np.mean(tvds)) if tvds else float("nan")


# ----------------------------------------------------------------------------
# 3. Correlation-matrix difference (Frobenius)
# ----------------------------------------------------------------------------
def corr_diff(real, synth, ds_cfg) -> float:              # lower better
    numeric, _, _ = _types(real, ds_cfg)
    if len(numeric) < 2:
        return float("nan")
    rc = real[numeric].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0).values
    sc = synth[numeric].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0).values
    return float(np.linalg.norm(rc - sc, ord="fro"))


# ----------------------------------------------------------------------------
# 4. Classifier two-sample test (AUC; closer to 0.5 is better)
# ----------------------------------------------------------------------------
def c2st_auc(real, synth, ds_cfg, seed: int = 0) -> float:
    numeric, categorical, _ = _types(real, ds_cfg)
    real, synth = real.copy(), synth[real.columns].copy()
    X = pd.concat([real, synth], ignore_index=True)
    y = np.r_[np.zeros(len(real)), np.ones(len(synth))]
    pipe = Pipeline([
        ("prep", _build_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(auc))


# ----------------------------------------------------------------------------
# 5. Train-on-Synthetic, Test-on-Real (ROC-AUC; higher better)
# ----------------------------------------------------------------------------
def tstr(real_train, synth, real_holdout, ds_cfg) -> float:
    """Fit HistGradientBoostingClassifier on synth, evaluate ROC-AUC on the real
    holdout. `real_train` is unused for scoring but kept for signature symmetry
    (the real baseline is computed separately by the runner)."""
    target = ds_cfg["target"]
    drop = set(ds_cfg.get("drop_for_tstr", [])) | {target}
    feats = [c for c in real_train.columns if c not in drop]
    numeric = [c for c in feats if c in _types(real_train, ds_cfg)[0]]
    categorical = [c for c in feats if c in _types(real_train, ds_cfg)[1]]
    pos = ds_cfg.get("positive_class", 1)

    ytr = (synth[target].astype(str) == str(pos)).astype(int).values
    yte = (real_holdout[target].astype(str) == str(pos)).astype(int).values
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")

    pipe = Pipeline([
        ("prep", _build_preprocessor(numeric, categorical)),
        ("clf", HistGradientBoostingClassifier(random_state=0)),
    ])
    from sklearn.metrics import roc_auc_score
    pipe.fit(synth[feats], ytr)
    proba = pipe.predict_proba(real_holdout[feats])[:, 1]
    return float(roc_auc_score(yte, proba))


# ----------------------------------------------------------------------------
# Convenience: compute the 4 distribution metrics in one call (TSTR needs holdout)
# ----------------------------------------------------------------------------
def _align_synth_dtypes(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """Cast synth columns to match real's dtype for INTEGER columns. LLM generators
    (notably GReaT) often emit an integer-coded column as a float ('1.0'), which makes
    the string-based categorical metric (TVD) and the TSTR target match treat '1.0' != '1'
    -> a spurious TVD=1.0 / single-class TSTR=NaN. Round + nullable-int cast fixes it
    without touching genuinely continuous columns. (the project spec 12.9: realign before metrics.)"""
    out = synth.copy()
    for c in real.columns:
        if c in out.columns and pd.api.types.is_integer_dtype(real[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    return out


def compute_all(real_train, synth, real_holdout, ds_cfg, seed: int = 0) -> dict:
    synth = _align_synth_dtypes(real_train, synth)
    return {
        "ks_marginal": ks_marginal(real_train, synth, ds_cfg),
        "tvd_categorical": tvd_categorical(real_train, synth, ds_cfg),
        "corr_diff": corr_diff(real_train, synth, ds_cfg),
        "c2st_auc": c2st_auc(real_train, synth, ds_cfg, seed=seed),
        "tstr": tstr(real_train, synth, real_holdout, ds_cfg),
    }
