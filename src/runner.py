"""Experiment loop: dataset x generator x config x seed.

Two-phase design (the project spec sec 12.3):
  Phase A (generation) — each cell runs as a subprocess in its generator's OWN env
    (env-llm / env-sdv) and writes generated.csv + gen_meta.json. Decouples the
    conflicting dependency stacks.
  Phase B (evaluation) — a single pass in the NEUTRAL env reads cached CSVs and
    computes the 5 metrics into metrics.json.

HARD RULES:
- Cache key = cell dir results/cells/{dataset}__{generator}__{config_id}__seed{n};
  the config HASH is stored in meta and checked on resume (stale config => recompute).
- Controls run ONLY at config 'baseline' (config-invariant).
- Never recompute a cached, hash-matching cell. Resume freely.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import configs, data, metrics

ROOT = Path(__file__).resolve().parent.parent


def _env_python(env_name: str) -> str:
    """Interpreter for a generator's env. On the Mac we keep two isolated venvs
    (env-llm / env-sdv) because sdv and be-great conflict (the project spec sec 12.3). In a
    single-env setup (e.g. the GPU Docker image or one shared venv) those subdirs
    don't exist -> fall back to the current interpreter so the same runner works
    everywhere. Override per env with ENV_LLM_PYTHON / ENV_SDV_PYTHON if needed."""
    override = os.environ.get(env_name.replace("-", "_").upper() + "_PYTHON")
    if override:
        return override
    cand = ROOT / env_name / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for m in ("pandas", "numpy", "scipy", "sklearn", "statsmodels"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def apply_smoke(cfg: dict) -> dict:
    """Shrink the grid for a fast sanity run (the project spec sec 10 step 6):
    1 dataset x all gens x 2 configs x 2 seeds, tiny n & epochs."""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["datasets"] = [d for d in cfg["datasets"] if d["name"] == "diabetes"]
    cfg["seeds"] = [0, 1]
    cfg["sample_size"]["n"] = 60
    cfg["sample_size"]["great_finetune_max_rows"] = 200
    cfg["_smoke_config_ids"] = ["baseline", "numeric_format=rounded"]
    for g in cfg["generators"]:
        if g["name"] == "great":
            g["params"].update(epochs=2, batch_size=8)
        elif g["name"] in ("ctgan", "tvae"):
            g["params"].update(epochs=10)
        elif g["name"] == "icl":
            g["params"].update(chunk_size=10, max_oversample=2, n_shots=10)
    return cfg


def _configs_for(cfg: dict, gen_cfg: dict) -> list[dict]:
    all_confs = configs.enumerate_configs(cfg["config"])
    if not gen_cfg.get("config_sensitive", False):
        return [c for c in all_confs if c["id"] == "baseline"]
    allow = cfg.get("_smoke_config_ids")
    return [c for c in all_confs if allow is None or c["id"] in allow]


def run_all(cfg: dict, smoke: bool = False, only_generators: list[str] | None = None) -> None:
    if smoke:
        cfg = apply_smoke(cfg)
    if only_generators:
        keep = set(only_generators)
        cfg["generators"] = [g for g in cfg["generators"] if g["name"] in keep]
        print(f"[runner] generator filter -> {[g['name'] for g in cfg['generators']]}")

    results_dir = Path(cfg["paths"]["results_dir"])
    cells_dir = results_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / ".effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (results_dir / "env.json").write_text(json.dumps(
        {"neutral_env": _versions(), "smoke": smoke}, indent=2))

    # --- prepare data + fixed splits (neutral env) ---
    data.ensure_datasets(cfg["paths"]["data_dir"])
    splits: dict[str, tuple] = {}
    for ds in cfg["datasets"]:
        name = ds["name"]
        df = data.load_dataset(name, cfg)
        dsc = data._ds_cfg(name, cfg)
        train, holdout = data.fixed_split(df, dsc["target"], name=name,
                                          results_dir=results_dir)
        n_gen = min(int(cfg["sample_size"]["n"]), len(train))
        splits[name] = (train, holdout, dsc, n_gen)

    # --- build the cell list ---
    cells = []
    for ds in cfg["datasets"]:
        name = ds["name"]
        _, _, _, n_gen = splits[name]
        for gen_cfg in cfg["generators"]:
            for conf in _configs_for(cfg, gen_cfg):
                for seed in cfg["seeds"]:
                    cid = conf["id"]
                    cell = cells_dir / f"{name}__{gen_cfg['name']}__{cid}__seed{seed}"
                    cells.append((name, gen_cfg, conf, seed, n_gen, cell))

    # --- Phase A: generation (subprocess per cell, in the gen's env) ---
    print(f"[runner] Phase A: {len(cells)} generation cells")
    for name, gen_cfg, conf, seed, n_gen, cell in cells:
        gm = cell / "gen_meta.json"
        # Freshness needs BOTH the config hash AND the requested row count to match.
        # config_hash covers only serialization/order/format, NOT n — so without the
        # n check a smoke cell (n=60) would be silently reused for the full grid
        # (n=5000), contaminating real results. (the project spec sec 8 & 12.9.)
        fresh = _hash_ok(gm, conf["hash"]) and _meta_n(gm) == n_gen
        if (cell / "generated.csv").exists() and fresh:
            print(f"  skip(gen)  {cell.name}")
            continue
        if gm.exists() and not (cell / "generated.csv").exists() and fresh:
            # previously errored with matching config+n; don't retry automatically
            print(f"  skip(err)  {cell.name}  ({_status(gm)})")
            continue
        # Stale or new: drop any old metrics so Phase B recomputes on the fresh csv.
        (cell / "metrics.json").unlink(missing_ok=True)
        print(f"  gen        {cell.name}  [{gen_cfg['env']}]")
        cmd = [
            _env_python(gen_cfg["env"]), "-m", "src.generate_one",
            "--config", str(results_dir / ".effective_config.yaml"),
            "--results-dir", str(results_dir),
            "--dataset", name, "--generator", gen_cfg["name"],
            "--config-id", conf["id"], "--seed", str(seed),
            "--n", str(n_gen), "--cell", str(cell),
        ]
        env = _subenv(gen_cfg, results_dir)
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if not gm.exists():
            # Hard crash (e.g. a native abort) before the subprocess could record
            # its own meta. Stamp a stub so the cell is visible and resume is sane.
            cell.mkdir(parents=True, exist_ok=True)
            gm.write_text(json.dumps({
                "status": "crashed", "config_hash": conf["hash"],
                "dataset": name, "generator": gen_cfg["name"],
                "config_id": conf["id"], "seed": seed, "n_requested": n_gen,
                "error": f"subprocess exited {proc.returncode} without writing gen_meta",
            }, indent=2))
        print(f"             -> {_status(gm)}")

    # --- Phase B: evaluation (neutral env) ---
    print(f"[runner] Phase B: evaluating metrics")
    for name, gen_cfg, conf, seed, n_gen, cell in cells:
        gpath = cell / "generated.csv"
        mpath = cell / "metrics.json"
        if not gpath.exists():
            continue
        if mpath.exists() and _hash_ok(mpath, conf["hash"]):
            continue
        # One pathological cell (degenerate synth, single-class target, NaN, ...) must
        # NOT abort the whole metrics pass after a long generation run. Isolate failures:
        # record a metrics_error.json (NOT metrics.json, so analysis skips it and a later
        # run retries) and continue.
        try:
            train, holdout, dsc, _ = splits[name]
            synth = pd.read_csv(gpath)
            m = metrics.compute_all(train, synth, holdout, dsc, seed=seed)
            m.update({"dataset": name, "generator": gen_cfg["name"],
                      "config_id": conf["id"], "config_hash": conf["hash"], "seed": seed})
            mpath.write_text(json.dumps(m, indent=2))
            (cell / "metrics_error.json").unlink(missing_ok=True)
            print(f"  eval       {cell.name}  ks={m['ks_marginal']:.3f} c2st={m['c2st_auc']:.3f}")
        except Exception as e:
            (cell / "metrics_error.json").write_text(json.dumps({
                "status": "metrics_error", "error": f"{type(e).__name__}: {e}",
                "dataset": name, "generator": gen_cfg["name"],
                "config_id": conf["id"], "config_hash": conf["hash"], "seed": seed,
            }, indent=2))
            print(f"  eval  ERR  {cell.name}  {type(e).__name__}: {e}")

    print("[runner] done.")


def _subenv(gen_cfg: dict, results_dir: Path) -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["HF_HOME"] = str((results_dir / ".model_cache").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _hash_ok(meta_path: Path, expected_hash: str) -> bool:
    return _read_json(meta_path).get("config_hash") == expected_hash


def _meta_n(meta_path: Path):
    """Requested row count recorded in a cell's gen_meta (None if absent). Used to
    detect smoke(n=60)-vs-full(n=5000) staleness that config_hash alone misses."""
    return _read_json(meta_path).get("n_requested")


def _status(meta_path: Path) -> str:
    return _read_json(meta_path).get("status", "?")
