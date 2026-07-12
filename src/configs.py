"""Task-irrelevant configuration transforms (the independent variable).

These change ONLY how a row is presented to an LLM generator, never the underlying
data or the task. An ideal generator would be invariant to all of them.

Used by LLM generators. Control generators (CTGAN/TVAE) MUST ignore configs entirely.

OFAT design (see config.yaml): 7 config points total.
Each config has an id like 'baseline', 'serialization=keyvalue', 'column_order=perm2', etc.

IMPORTANT (the project spec sec 3): config and seed are SEPARATE axes. The column-order
permutations and numeric rounding are therefore derived deterministically from the
config id alone, so a given config is byte-identical across all 5 run seeds.
"""
from __future__ import annotations

import hashlib
import json
from math import floor, log10

import numpy as np
import pandas as pd

# Fixed base for column-order permutations. NOT the run seed (see module docstring).
PERM_BASE_SEED = 12345
_PERM_INDEX = {"perm1": 1, "perm2": 2, "perm3": 3}


# ----------------------------------------------------------------------------
# Config enumeration
# ----------------------------------------------------------------------------
def enumerate_configs(config_cfg: dict) -> list[dict]:
    """Expand baseline + OFAT axes into the 7 frozen config dicts, each with an 'id'
    and a stable content 'hash' (for the cache key)."""
    base = config_cfg["baseline"]
    axes = config_cfg["axes"]
    configs: list[dict] = [dict(base, id="baseline")]

    for scheme in axes["serialization"]:
        configs.append(dict(base, serialization=scheme, id=f"serialization={scheme}"))
    for order in axes["column_order"]:
        configs.append(dict(base, column_order=order, id=f"column_order={order}"))
    for fmt in axes["numeric_format"]:
        configs.append(dict(base, numeric_format=fmt, id=f"numeric_format={fmt}"))

    for c in configs:
        c["hash"] = config_hash(c)
    return configs


def config_hash(config: dict) -> str:
    """Stable short hash over the meaningful config fields (id excluded). Used in the
    cache key so a changed config.yaml never silently reuses a stale cell."""
    payload = {k: config[k] for k in ("serialization", "column_order", "numeric_format")}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:10]


# ----------------------------------------------------------------------------
# Numeric formatting
# ----------------------------------------------------------------------------
def _round_sig(x: float, sig: int = 2) -> float:
    if x == 0 or not np.isfinite(x):
        return float(x)
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def _fmt_value(val, numeric_format: str) -> str:
    """Render a single cell to text. 'rounded' -> continuous (float) values to 2 sig
    figs; integer-valued cells render without a decimal point. Strings pass through."""
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        v = _round_sig(float(val), 2) if numeric_format == "rounded" else float(val)
        return str(int(v)) if float(v).is_integer() else repr(v)
    return str(val)


def format_numbers(df: pd.DataFrame, numeric_format: str) -> pd.DataFrame:
    """Return a copy with continuous (float) columns rounded to 2 sig figs when
    numeric_format=='rounded'. Integer columns are untouched. 'raw' is a no-op."""
    if numeric_format != "rounded":
        return df
    out = df.copy()
    for c in out.select_dtypes(include="float").columns:
        out[c] = out[c].map(lambda x: _round_sig(float(x), 2) if pd.notna(x) else x)
    return out


# ----------------------------------------------------------------------------
# Column order
# ----------------------------------------------------------------------------
def apply_column_order(df: pd.DataFrame, order_id: str, seed: int = PERM_BASE_SEED) -> pd.DataFrame:
    """order_id: 'original' | 'perm1'|'perm2'|'perm3'.
    Permutations are deterministic in (order_id, seed) and INDEPENDENT of the run
    seed, so the same config presents identical column orders across all run seeds.
    `seed` defaults to PERM_BASE_SEED; callers should not pass a run seed here."""
    if order_id == "original":
        return df
    if order_id not in _PERM_INDEX:
        raise ValueError(f"unknown column_order {order_id!r}")
    rng = np.random.default_rng(seed + _PERM_INDEX[order_id])
    perm = rng.permutation(df.shape[1])
    return df.iloc[:, perm]


# ----------------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------------
def serialize_row(row: pd.Series, scheme: str, numeric_format: str) -> str:
    """Row -> text string.
    scheme: 'sentence' | 'keyvalue' | 'compact'
      sentence : "The age is 35. The income is 50000."
      keyvalue : "age=35, income=50000"
      compact  : "35|50000" (header given separately)
    numeric_format: 'raw' | 'rounded'
    """
    pairs = [(col, _fmt_value(val, numeric_format)) for col, val in row.items()]
    if scheme == "sentence":
        return " ".join(f"The {col} is {val}." for col, val in pairs)
    if scheme == "keyvalue":
        return ", ".join(f"{col}={val}" for col, val in pairs)
    if scheme == "compact":
        return "|".join(val for _, val in pairs)
    raise ValueError(f"unknown serialization scheme {scheme!r}")


def serialize_frame(df: pd.DataFrame, config: dict) -> tuple[str | None, list[str]]:
    """Apply column order + numeric format, then serialize every row per the scheme.
    Returns (header, row_texts). `header` is the '|'-joined column line for the
    'compact' scheme (which omits column names per row) and None otherwise.
    Used by the ICL generator and to drive GReaT's serializer.

    Rounding is done HERE at the frame level (dtype-aware: only float columns), then
    rows are rendered with numeric_format='raw'. This is why integer columns stay
    integer even under 'rounded' — iterrows() would otherwise coerce a mixed row to
    float and lose that distinction (see serialize_row's limitation)."""
    ordered = apply_column_order(df, config["column_order"])
    presented = format_numbers(ordered, config["numeric_format"])
    scheme = config["serialization"]
    rows = [serialize_row(r, scheme, "raw") for _, r in presented.iterrows()]
    header = "|".join(presented.columns) if scheme == "compact" else None
    return header, rows
