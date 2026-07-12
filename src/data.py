"""Data loading, preprocessing, and ONE fixed train/holdout split per dataset.

Scope rules (see the project spec):
- Exactly the 3 datasets in config.yaml. No others.
- Split is made ONCE with a fixed seed and reused everywhere (do not re-split per run).
- `student` target is derived: pass = 1 if G3 >= 10 else 0 (Portuguese 0-20 grading).

Determinism (the project spec sec 12.2):
- Each raw source is pinned by URL + md5; shape is asserted on load.
- Files live in data/ (git-ignored). `ensure_datasets()` re-downloads if missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Frozen dataset registry. md5 / shapes pinned from the deterministic downloads.
# ----------------------------------------------------------------------------
HOLDOUT_FRAC = 0.20          # ONE fixed split; holdout is the real-test set for TSTR
SPLIT_SEED = 42              # independent of run seed; the split is made once

# Canonical column names for the headerless sources.
_ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
_DIABETES_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

DATASET_SPECS: dict[str, dict] = {
    "adult": {
        "raw_file": "adult.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "md5": "5d7c39d7b8804f071cdd1f2a7c460872",
        "expected_shape": (32561, 15),   # after dropping the trailing blank line
        "target": "income",
        "positive_class": ">50K",
    },
    "diabetes": {
        "raw_file": "diabetes_raw.csv",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "md5": "56a8d8ae619fcc223941e54f361b8406",
        "expected_shape": (768, 9),
        "target": "Outcome",
        "positive_class": 1,
    },
    "student": {
        "raw_file": "student-mat.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",  # student-mat.csv inside
        "md5": "4dc304be95c60de6ee13fb8769469dd7",
        "expected_shape": (395, 33),     # raw, before deriving `pass`
        "target": "pass",
        "positive_class": 1,
        "drop_for_tstr": ["G1", "G2", "G3"],
    },
}


def _md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_datasets(data_dir: str | Path) -> None:
    """Download any missing raw file deterministically. Existing files are left
    untouched (md5 is verified at load time, not here)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in DATASET_SPECS.items():
        dest = data_dir / spec["raw_file"]
        if dest.exists():
            continue
        url = spec["url"]
        if name == "student":
            import io
            import zipfile
            with urllib.request.urlopen(url) as resp:
                z = zipfile.ZipFile(io.BytesIO(resp.read()))
            dest.write_bytes(z.read("student-mat.csv"))
        else:
            urllib.request.urlretrieve(url, dest)


def _read_raw(name: str, data_dir: Path) -> pd.DataFrame:
    """Read the pinned raw file, verifying md5, into a DataFrame (no cleaning yet)."""
    spec = DATASET_SPECS[name]
    path = data_dir / spec["raw_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run data.ensure_datasets({data_dir!r}) first."
        )
    got = _md5(path)
    if got != spec["md5"]:
        raise ValueError(
            f"{name}: md5 mismatch for {path.name}. expected {spec['md5']}, got {got}. "
            "The source changed — do not proceed (frozen study)."
        )

    if name == "adult":
        df = pd.read_csv(
            path, header=None, names=_ADULT_COLS, skipinitialspace=True,
            na_values="?", skip_blank_lines=True,
        )
    elif name == "diabetes":
        df = pd.read_csv(path, header=None, names=_DIABETES_COLS)
    elif name == "student":
        df = pd.read_csv(path, sep=";")
    else:  # pragma: no cover - registry is frozen
        raise KeyError(name)
    return df


def load_dataset(name: str, cfg: dict) -> pd.DataFrame:
    """Load + clean a dataset; verify md5 and shape. Returns a clean DataFrame
    whose target column is named per DATASET_SPECS (derived for `student`)."""
    if name not in DATASET_SPECS:
        raise KeyError(f"{name} not in frozen registry {list(DATASET_SPECS)}")
    data_dir = Path(cfg["paths"]["data_dir"])
    spec = DATASET_SPECS[name]

    df = _read_raw(name, data_dir)
    exp_rows, exp_cols = spec["expected_shape"]
    if df.shape != (exp_rows, exp_cols):
        raise ValueError(
            f"{name}: raw shape {df.shape} != expected {(exp_rows, exp_cols)}"
        )

    df = preprocess(df, _ds_cfg(name, cfg))
    return df


def _ds_cfg(name: str, cfg: dict) -> dict:
    """Pull the per-dataset block from config.yaml, merged with frozen spec defaults."""
    block = next((d for d in cfg["datasets"] if d["name"] == name), {})
    spec = DATASET_SPECS[name]
    merged = {
        "name": name,
        "target": spec["target"],
        "positive_class": spec["positive_class"],
        "drop_for_tstr": spec.get("drop_for_tstr", []),
    }
    merged.update({k: v for k, v in block.items() if k not in merged or v is not None})
    return merged


def preprocess(df: pd.DataFrame, ds_cfg: dict) -> pd.DataFrame:
    """Light, generator-agnostic cleaning only. No feature engineering."""
    df = df.copy()
    name = ds_cfg["name"]

    # Strip stray whitespace from object columns (UCI files pad with spaces).
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    if name == "student":
        # Derive binary target; keep G1/G2/G3 (dropped only as TSTR predictors).
        df["pass"] = (df["G3"] >= 10).astype(int)

    if name == "adult":
        # Drop rows with missing values (the only dataset with NaNs).
        df = df.dropna(axis=0).reset_index(drop=True)

    # Move target to the last column for a stable canonical order.
    target = ds_cfg["target"]
    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols].reset_index(drop=True)
    return df


def fixed_split(df: pd.DataFrame, target: str, seed: int = SPLIT_SEED,
                results_dir: str | Path | None = None, name: str | None = None):
    """Return (train_df, holdout_df). Made ONCE per dataset (stratified, SPLIT_SEED);
    cached to results/splits/<name>/ and reused. `seed` here is the SPLIT seed, NOT
    a run seed — the split is identical across all runs."""
    cache = None
    if results_dir is not None and name is not None:
        cache = Path(results_dir) / "splits" / name
    if cache is not None and (cache / "train.csv").exists():
        train = pd.read_csv(cache / "train.csv")
        holdout = pd.read_csv(cache / "holdout.csv")
        return train, holdout

    stratify = df[target] if df[target].nunique() > 1 else None
    train, holdout = train_test_split(
        df, test_size=HOLDOUT_FRAC, random_state=seed, stratify=stratify,
    )
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        train.to_csv(cache / "train.csv", index=False)
        holdout.to_csv(cache / "holdout.csv", index=False)
        (cache / "split_meta.json").write_text(json.dumps({
            "name": name, "target": target, "split_seed": seed,
            "holdout_frac": HOLDOUT_FRAC, "n_train": len(train), "n_holdout": len(holdout),
        }, indent=2))
    return train, holdout


def column_types(df: pd.DataFrame, target: str | None = None) -> dict:
    """Return {'numeric': [...], 'categorical': [...], 'target': target}.

    Categorical = object/category/bool dtype columns plus the (classification) target.
    Numeric = the remaining numeric-dtype columns. Generator-agnostic; used by both
    serialization and metrics.
    """
    categorical, numeric = [], []
    for c in df.columns:
        # is_numeric_dtype (not `dtype == object`) so a pandas that reads strings as the
        # newer 'string' dtype doesn't mis-type categorical columns as numeric.
        if (target is not None and c == target) or not pd.api.types.is_numeric_dtype(df[c]):
            categorical.append(c)
        else:
            numeric.append(c)
    return {"numeric": numeric, "categorical": categorical, "target": target}
