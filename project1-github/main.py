from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def check():
    required = [ROOT / "requirements.txt", ROOT / "README.md", ROOT / "dataset_description.md",
                ROOT / "results/classification_model_comparison.csv", ROOT / "weights"]
    missing = [str(path) for path in required if not path.exists()]
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Required project files: {'OK' if not missing else 'MISSING'}")
    for path in missing: print(f"  - {path}")
    data = ROOT / "data"
    print(f"Dataset directory: {'available' if data.exists() else 'not bundled; see dataset_description.md'}")
    return 0 if not missing else 1


def main():
    parser = argparse.ArgumentParser(description="HAM10000 Final Project unified entry point")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("--models", nargs="+", default=["EfficientNetB0"])
    evaluate.add_argument("--data-dir", type=Path, default=ROOT / "data"); evaluate.add_argument("--batch-size", type=int, default=16)
    segment = commands.add_parser("segment"); segment.add_argument("--data-dir", type=Path, default=ROOT / "data")
    segment.add_argument("--epochs", type=int, default=5); segment.add_argument("--batch-size", type=int, default=32); segment.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if args.command == "check": raise SystemExit(check())
    if args.command == "evaluate":
        command = [sys.executable, str(ROOT / "source_code/evaluation/evaluate.py"), "--models", *args.models,
                   "--data-dir", str(args.data_dir), "--weights-dir", str(ROOT / "weights"),
                   "--output-dir", str(ROOT / "results/recomputed"), "--batch-size", str(args.batch_size)]
    else:
        command = [sys.executable, str(ROOT / "source_code/segmentation/train.py"), "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size), "--size", str(args.size), "--data-dir", str(args.data_dir)]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
