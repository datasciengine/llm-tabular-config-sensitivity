"""Control generators: CTGAN, TVAE via SDV. CONFIG-INVARIANT BY DESIGN.

These never see serialized text. They take ONLY a seed. Their seed-variance is
the null/floor against which LLM config-variance is measured. Do NOT let config
leak in here -- if config != baseline, ignore it (and warn).
"""
from __future__ import annotations

import warnings

import pandas as pd

from .base import BaseGenerator, set_all_seeds


def _use_cuda() -> bool:
    """CTGAN/TVAE train fast on a GPU and are painfully slow on CPU for large
    datasets (adult: ~24k rows x 300 epochs). Auto-enable CUDA when present so the
    GPU run is fast; stays False on the Mac (no CUDA) -> CPU as before."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class _SDVControl(BaseGenerator):
    config_sensitive = False

    def _build(self, metadata, epochs):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, train_df, seed, config, ds_cfg):
        # Config-invariance guard: controls must only ever see baseline.
        if config is not None and config.get("id", "baseline") != "baseline":
            warnings.warn(
                f"{type(self).__name__} is config-invariant; ignoring config "
                f"{config.get('id')!r} (the project spec sec 3)."
            )
        set_all_seeds(seed)
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_df)
        epochs = int(self.params.get("epochs", 300))
        self._columns = list(train_df.columns)
        self.model = self._build(metadata, epochs)
        self.model.fit(train_df)

    def sample(self, n, seed):
        set_all_seeds(seed)
        df = self.model.sample(num_rows=n)
        return df[self._columns]   # canonical order for metrics


class CTGANGenerator(_SDVControl):
    def _build(self, metadata, epochs):
        from sdv.single_table import CTGANSynthesizer
        return CTGANSynthesizer(metadata, epochs=epochs, cuda=_use_cuda(), verbose=False)


class TVAEGenerator(_SDVControl):
    def _build(self, metadata, epochs):
        from sdv.single_table import TVAESynthesizer
        return TVAESynthesizer(metadata, epochs=epochs, cuda=_use_cuda())
