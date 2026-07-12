"""GReaT adapter (Borisov et al.). Install: pip install be-great.

Re-serializes training data per `config` (serialization/order/format) BEFORE
fine-tuning the small base model (distilgpt2).

How each config axis reaches GReaT
----------------------------------
- numeric_format : applied to the dataframe (configs.format_numbers) before fit.
- column_order   : applied to the dataframe before fit. NOTE GReaT also permutes
                   columns internally each row (its documented augmentation), so it
                   is largely order-invariant by design — an honest finding, kept
                   per the project spec ("use documented defaults").
- serialization  : 'sentence' is realized by GReaT's NATIVE "<col> is <val>"
                   template (the format its parser AND start-sampler are built
                   around). 'keyvalue'/'compact' require overriding GReaT's coupled
                   encode + decode + start-sampler; that is the open design decision
                   flagged to the human (the project spec sec 12 did not resolve it), so
                   those two schemes are gated below until the approach is ratified.
"""
from __future__ import annotations

import os

import pandas as pd

from .base import BaseGenerator, set_all_seeds
from .. import configs

_NATIVE_SCHEME = "sentence"   # GReaT's native template stands in for 'sentence'


class GReaTGenerator(BaseGenerator):
    config_sensitive = True

    def fit(self, train_df, seed, config, ds_cfg):
        set_all_seeds(seed)
        scheme = config["serialization"]
        if scheme != _NATIVE_SCHEME:
            raise NotImplementedError(
                f"GReaT serialization={scheme!r} is pending a design decision "
                "(override of GReaT's coupled encode/decode/start-sampler + the "
                "column-shuffle interplay). See great.py docstring / the project spec sec 12. "
                "Only serialization='sentence' (GReaT native) is implemented."
            )

        self.config = config
        self.ds_cfg = ds_cfg
        self.columns = list(train_df.columns)               # canonical order
        self._dtypes = {c: train_df[c].dtype for c in train_df.columns}

        df = configs.apply_column_order(train_df, config["column_order"])
        df = configs.format_numbers(df, config["numeric_format"])

        from be_great import GReaT

        epochs = int(self.params.get("epochs", 50))
        batch_size = int(self.params.get("batch_size", 32))
        exp_dir = os.environ.get("GREAT_EXP_DIR", "results/.great_trainer")
        self.model = GReaT(
            self.base_model or "distilgpt2",
            experiment_dir=exp_dir,
            epochs=epochs,
            batch_size=batch_size,
            report_to=[],
            save_strategy="no",
            logging_strategy="no",
            fp16=False,
            dataloader_num_workers=0,
            seed=seed,
        )
        self.model.fit(df)

    def sample(self, n, seed):
        set_all_seeds(seed)
        import torch

        device = (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        df = self.model.sample(n_samples=n, device=device, max_length=200)
        return self._postprocess(df, n)

    def _postprocess(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Coerce GReaT output back to the canonical schema: keep known columns,
        coerce numeric dtypes, drop unparseable rows, realign column order, pad to n."""
        for c in self.columns:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[self.columns].copy()
        for c in self.columns:
            if pd.api.types.is_numeric_dtype(self._dtypes[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=0).reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("GReaT produced 0 parseable rows for this cell.")
        if len(df) < n:
            df = df.sample(n=n, replace=True, random_state=0).reset_index(drop=True)
        return df.iloc[:n].reset_index(drop=True)
