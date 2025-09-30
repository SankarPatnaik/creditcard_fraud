#!/usr/bin/env python3
"""Switch the active dataset between the tiny synthetic CSV and a Kaggle CSV.

Usage:
  python scripts/switch_dataset.py --use tiny
  python scripts/switch_dataset.py --use kaggle --kaggle-path /path/to/creditcard.csv

Effect:
  Copies the chosen CSV to data/raw/transactions.csv so all code uses a consistent path.
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ACTIVE = RAW_DIR / "transactions.csv"  # canonical path used by notebooks/scripts

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use", choices=["tiny", "kaggle"], required=True)
    p.add_argument("--kaggle-path", type=str, help="Path to Kaggle creditcard.csv file (local)")
    return p.parse_args()

def main():
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.use == "tiny":
        src = RAW_DIR / "tiny_transactions.csv"
        if not src.exists():
            print(f"[ERROR] Tiny dataset not found at {src}", file=sys.stderr)
            sys.exit(1)
        shutil.copyfile(src, ACTIVE)
        print(f"[OK] Switched to tiny dataset -> {ACTIVE}")
    else:
        if not args.kaggle_path:
            print("[ERROR] --kaggle-path is required when --use kaggle", file=sys.stderr)
            sys.exit(1)
        src = Path(args.kaggle_path).expanduser().resolve()
        if not src.exists():
            print(f"[ERROR] Kaggle CSV not found at {src}", file=sys.stderr)
            sys.exit(1)
        shutil.copyfile(src, ACTIVE)
        print(f"[OK] Switched to Kaggle dataset -> {ACTIVE}")

if __name__ == "__main__":
    main()
