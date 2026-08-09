"""Command-line entry point: train the CvT-TCN model and report test accuracy.

Examples:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml --epochs 120
"""

import argparse

from cvttcn.config import Config, load_config
from cvttcn.data.preprocessing import load_or_build_epochs
from cvttcn.experiment import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CvT-TCN EEG motor-imagery model."
    )
    parser.add_argument(
        "--config", default=None, help="YAML config path (defaults to built-in defaults)."
    )
    parser.add_argument(
        "--cache", default="data/epochs_all.npz", help="Cached epochs .npz path."
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory for checkpoints, plots, results."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the number of training epochs."
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Rebuild the epoch cache from raw EDF."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else Config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    cfg.validate()

    data = load_or_build_epochs(cfg.data, args.cache, force=args.force_rebuild)
    result = run_training(cfg, data, args.output_dir)

    summary = result["summary"]
    print(
        f"test accuracy: {summary['test_accuracy']:.4f} | "
        f"macro-F1: {summary['test_macro_f1']:.4f} | "
        f"kappa: {summary['test_kappa']:.4f}"
    )
    print("PASS: cleared the 70% target" if summary["test_accuracy"] > 0.70 else "below 70% target")


if __name__ == "__main__":
    main()
