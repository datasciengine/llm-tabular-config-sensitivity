# Configuration Sensitivity of LLM-Based Synthetic Tabular Data Generators

[![Published in IEEE Access](https://img.shields.io/badge/Published%20in-IEEE%20Access%20(2026)-00629B)](https://doi.org/10.1109/ACCESS.2026.3732216)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3732216-blue)](https://doi.org/10.1109/ACCESS.2026.3732216)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Published in *IEEE Access*, 2026.** Şahin, Varlıklar, and Kılınç,
> *"Configuration Sensitivity of LLM Tabular Data Synthesizers: A Small-Model
> Memorization Effect and a Reporting Protocol."*
> DOI: [10.1109/ACCESS.2026.3732216](https://doi.org/10.1109/ACCESS.2026.3732216) (Open Access, CC BY 4.0).
> Please [cite the paper](#citation) if you use this code.

This repository contains the code and experimental configuration for an empirical
study of how **task-irrelevant configuration choices** — the serialization template,
column order, and numeric formatting used to present tabular rows to a language model —
affect the **fidelity and utility** of LLM-based synthetic tabular data generators,
and whether such choices **change the ranking** of generators.

Non-LLM generators (CTGAN, TVAE) never see serialized text and are config-invariant by
construction; their seed-only variance provides a noise floor against which the
LLM generators' configuration-induced variance is measured.

## Research questions
- **RQ1** — How much do task-irrelevant configuration changes move standard fidelity/utility metrics?
- **RQ2** — Do these changes flip the ranking of generators (i.e., are single-configuration comparisons safe)?
- **RQ3** — How is total variance distributed across {configuration, seed, generator, dataset}?

## Experiment matrix
`config.yaml` is the single source of truth for every setting below.

| Axis | Values |
|---|---|
| Datasets | UCI Adult, Pima Indians Diabetes, UCI Student Performance (Math) |
| LLM generators | `great` (fine-tuned, base `distilgpt2`), `icl` (few-shot, no training) |
| Control generators | `ctgan`, `tvae` (via SDV) |
| Configuration axes (LLM only, one-factor-at-a-time) | serialization ×3, column order ×4, numeric format ×2 → 7 points |
| Seeds | 5 (`0–4`) |
| Metrics | KS (numeric marginals), TVD (categorical marginals), correlation-matrix difference (Frobenius), classifier two-sample test AUC, train-on-synthetic/test-on-real ROC-AUC |

Total: 240 generation runs (LLM 105 + 105, controls 30), with the five metrics computed on each.

### Extended runs
Beyond the base grid above (`config.yaml`), two extensions back the main analysis:
- **Primary case study (`config_reseed.yaml`):** the diabetes cell is re-run at **15 seeds**
  for adequate statistical power (variance components, ranking stability, treatment
  validation, memorization/dedup, privacy).
- **Scale probe:** the in-context arm is additionally run with **`Qwen2.5-7B-Instruct`**
  across all three datasets to test how verbatim copying and valid-row yield change with
  model scale. (GReaT remains `distilgpt2`.)

The `scripts/` directory holds the analysis and figure code (variance decomposition with a
generator×configuration interaction, dedup with a size-matched control, treatment
validation against an independent reference, DCR/NNDR privacy indicators, classifier
sensitivity, and figure generation).

## Repository layout
```
config.yaml              # frozen settings (single source of truth)
src/data.py              # dataset loading, preprocessing, fixed train/holdout splits
src/configs.py           # serialization / column-order / numeric-format transforms
src/generators/          # base interface + great, icl, controls (CTGAN/TVAE)
src/metrics.py           # the five fidelity/utility metrics
src/runner.py            # dataset × generator × config × seed loop; caching + resume
src/analysis.py          # variance components, ranking stability, CSI, figures
scripts/run_all.py       # entry point
```

## Setup
The LLM and SDV stacks are installed in two isolated environments to avoid dependency
conflicts; analysis runs in a neutral environment.

```bash
# LLM generators (great, icl)
python3 -m venv env-llm && env-llm/bin/pip install transformers accelerate be-great torch
# SDV controls (ctgan, tvae)
python3 -m venv env-sdv && env-sdv/bin/pip install sdv
# analysis (neutral)
pip install pandas numpy scipy scikit-learn statsmodels pyyaml matplotlib
```

Raw datasets are placed under `data/` (git-ignored); `src/data.py` prepares them and
writes one fixed train/holdout split per dataset.

## Running
```bash
python scripts/run_all.py --smoke                     # small sanity run first
python scripts/run_all.py                             # full frozen grid (240 runs)
python scripts/run_all.py --generators ctgan,tvae     # optional: run a subset
```
The runner caches each generated table and its metrics under `results/` and resumes
without recomputation. After the grid completes:
```bash
python src/analysis.py --results-dir results          # tables + figures -> results/analysis/
```

## Reproducibility
- Every run takes an explicit integer seed; all randomness (Python, NumPy, PyTorch,
  generator-internal) is seeded.
- Library versions are logged to `results/env.json`.
- Metrics are always computed on the canonical column order; the column-order
  perturbation reaches only the model during fitting, never the evaluation.

## Statistics
Metrics are reported as mean ± 95% CI across seeds. Multiple comparisons are
Holm-corrected within separate families (configuration effects; generator-pair
rankings). Variance components are estimated with an ANOVA decomposition
(`statsmodels`), with a mixed-model cross-check.

## Citation

If you use this code or build on this work, please cite the paper:

```bibtex
@article{sahin2026configuration,
  author  = {{\c{S}}ahin, Murat and Varl{\i}klar, {\"O}zlem and K{\i}l{\i}n{\c{c}}, Deniz},
  title   = {Configuration Sensitivity of {LLM} Tabular Data Synthesizers:
             A Small-Model Memorization Effect and a Reporting Protocol},
  journal = {IEEE Access},
  year    = {2026},
  publisher = {IEEE},
  doi     = {10.1109/ACCESS.2026.3732216},
  url     = {https://doi.org/10.1109/ACCESS.2026.3732216}
}
```

The paper is Open Access under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
license; the code in this repository is released under the [MIT License](LICENSE).
