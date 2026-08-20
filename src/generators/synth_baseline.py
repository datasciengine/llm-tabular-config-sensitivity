"""Recent strong non-LLM baseline via synthcity (R2-6).

The reviewer asked for a stronger / more recent tabular-synthesis baseline than
CTGAN/TVAE. We add a diffusion-based synthesizer (TabDDPM, Kotelnikov et al. 2023)
through the `synthcity` plugin registry. Like the SDV controls it NEVER sees serialized
text, so it is CONFIG-INVARIANT BY DESIGN and contributes only seed variance — it slots
straight into the existing control-floor / ranking machinery.

The backend is swappable via params.plugin (default "ddpm" = TabDDPM). If TabDDPM proves
unstable on a given table, a robust fallback (e.g. "arf", "tvae") can be selected from
config WITHOUT a code change.
"""
from __future__ import annotations

import warnings

import pandas as pd

from .base import BaseGenerator, set_all_seeds


class SynthcityGenerator(BaseGenerator):
    """Config-invariant baseline backed by a synthcity plugin (default: TabDDPM)."""

    config_sensitive = False

    def fit(self, train_df, seed, config, ds_cfg):
        # Config-invariance guard: this baseline must only ever see the baseline config.
        if config is not None and config.get("id", "baseline") != "baseline":
            warnings.warn(
                f"{type(self).__name__} is config-invariant; ignoring config "
                f"{config.get('id')!r} (the project spec sec 3)."
            )
        set_all_seeds(seed)
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        self._columns = list(train_df.columns)
        self._dtypes = {c: train_df[c].dtype for c in train_df.columns}
        target = (ds_cfg or {}).get("target")

        plugin_name = str(self.params.get("plugin", "ddpm"))
        # Only pass hyper-params the plugin accepts; synthcity ignores unknown keys but we
        # keep the set small and explicit for reproducibility.
        kw = {}
        for k in ("n_iter", "batch_size", "lr", "num_timesteps", "n_iter_min",
                  "dim_embed", "random_state"):
            if k in self.params:
                kw[k] = self.params[k]
        kw.setdefault("random_state", int(seed))

        loader_kwargs = {}
        if target and target in train_df.columns:
            loader_kwargs["target_column"] = target
        self._loader_ref = GenericDataLoader(train_df, **loader_kwargs)

        self.model = Plugins().get(plugin_name, **kw)
        self.model.fit(self._loader_ref)

    def sample(self, n, seed):
        set_all_seeds(seed)
        # synthcity's generate() is stochastic; seed set above. Some plugins accept
        # random_state at generate time — pass it when supported, else fall back.
        try:
            out = self.model.generate(count=n, random_state=int(seed))
        except TypeError:
            out = self.model.generate(count=n)
        df = out.dataframe() if hasattr(out, "dataframe") else pd.DataFrame(out)
        # Realign to the canonical schema for metrics; coerce numeric dtypes back.
        for c in self._columns:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[self._columns].copy()
        for c in self._columns:
            if pd.api.types.is_numeric_dtype(self._dtypes[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=0).reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"{type(self).__name__} produced 0 parseable rows.")
        if len(df) < n:
            df = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
        return df.iloc[:n].reset_index(drop=True)
