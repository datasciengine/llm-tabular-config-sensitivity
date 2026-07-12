"""Entrypoint. Usage: python scripts/run_all.py [--smoke] [--generators icl,ctgan]"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path
import yaml
from src.runner import run_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="1 dataset x all gens x 2 configs x 2 seeds, tiny n")
    ap.add_argument("--generators", default=None,
                    help="comma-separated subset to run (e.g. 'icl,ctgan,tvae'); "
                         "omit to run all. Use to keep GReaT off the Mac (sec 12.1).")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gens = [g.strip() for g in args.generators.split(",")] if args.generators else None
    run_all(cfg, smoke=args.smoke, only_generators=gens)
