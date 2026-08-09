"""Command-line entry point: evaluate a trained checkpoint on the test split.

Example:
    python scripts/evaluate.py --checkpoint results/best_model.pt
"""

import argparse
from pathlib import Path

from cvttcn.config import Config, load_config
from cvttcn.data.dataset import build_dataloaders
from cvttcn.data.preprocessing import CLASS_NAMES, load_or_build_epochs
from cvttcn.models.cvt_tcn import build_model
from cvttcn.plots import plot_confusion
from cvttcn.training.metrics import compute_metrics
from cvttcn.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CvT-TCN checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--config", default=None, help="YAML config path (defaults to defaults).")
    parser.add_argument("--cache", default="data/epochs_all.npz", help="Cached epochs .npz path.")
    parser.add_argument("--output-dir", default="results", help="Where to write the plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else Config()

    # The seeded split is deterministic, so this reproduces the training test set.
    data = load_or_build_epochs(cfg.data, args.cache)
    loaders = build_dataloaders(data, cfg)

    trainer = Trainer(build_model(cfg), cfg)
    trainer.load_checkpoint(args.checkpoint)
    y_true, y_pred = trainer.predict(loaders.test)
    metrics = compute_metrics(y_true, y_pred, cfg.model.num_classes)

    names = (
        list(CLASS_NAMES)
        if cfg.model.num_classes == len(CLASS_NAMES)
        else [str(i) for i in range(cfg.model.num_classes)]
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(metrics.confusion, names, out_dir / "confusion_matrix_eval.png")

    print(
        f"test accuracy: {metrics.accuracy:.4f} | "
        f"macro-F1: {metrics.macro_f1:.4f} | kappa: {metrics.kappa:.4f}"
    )


if __name__ == "__main__":
    main()
