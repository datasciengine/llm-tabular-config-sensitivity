"""Common generator interface. Every generator implements fit() then sample()."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd


def set_all_seeds(seed: int) -> None:
    """Seed python / numpy / torch (+ MPS/CUDA) so a run is reproducible
    (the project spec sec 8: all randomness seeded)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


class BaseGenerator:
    config_sensitive: bool = False  # controls override to False; LLM gens True

    def __init__(self, params: dict, base_model: str | None = None):
        self.params = params or {}
        self.base_model = base_model

    def fit(self, train_df: pd.DataFrame, seed: int, config: dict, ds_cfg: dict) -> None:
        """Fit/prepare. LLM gens use `config` for serialization/order/format.
        Controls MUST ignore `config` (assert config is baseline or warn)."""
        raise NotImplementedError

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Return n synthetic rows with the SAME schema as train_df."""
        raise NotImplementedError


def make_generator(gen_cfg: dict) -> "BaseGenerator":
    """Factory: dispatch on gen_cfg['name']. Imports are lazy so env-sdv never
    needs be_great and env-llm never needs sdv (the project spec sec 12.3)."""
    name = gen_cfg["name"]
    params = gen_cfg.get("params", {})
    base_model = gen_cfg.get("base_model")
    if name == "great":
        from .great import GReaTGenerator
        return GReaTGenerator(params, base_model)
    if name == "icl":
        from .icl import ICLGenerator
        return ICLGenerator(params, base_model)
    if name == "ctgan":
        from .controls import CTGANGenerator
        return CTGANGenerator(params)
    if name == "tvae":
        from .controls import TVAEGenerator
        return TVAEGenerator(params)
    if name in ("tabddpm", "synthcity", "arf"):
        from .synth_baseline import SynthcityGenerator
        return SynthcityGenerator(params)
    raise ValueError(f"unknown generator {name!r}")
