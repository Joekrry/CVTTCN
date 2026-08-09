"""End-to-end training orchestration.

Ties the pieces together: split the pooled epochs into loaders, train with the
Trainer, evaluate on the held-out test split, and write artifacts (best
checkpoint, config, training-curve and confusion-matrix plots, and a results
JSON). Kept in the library (rather than the CLI script) so it can be exercised
by the integration test.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

from cvttcn.config import Config
from cvttcn.data.dataset import build_dataloaders
from cvttcn.data.preprocessing import CLASS_NAMES, EpochedData
from cvttcn.models.cvt_tcn import build_model
from cvttcn.plots import plot_confusion, plot_history
from cvttcn.training.metrics import compute_metrics
from cvttcn.training.trainer import Trainer
from cvttcn.training.utils import set_seed


def _class_names(num_classes: int) -> list[str]:
    if num_classes == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [str(i) for i in range(num_classes)]


def _history_to_jsonable(history: list[dict]) -> list[dict]:
    return [
        {
            "epoch": h["epoch"],
            "lr": h["lr"],
            "train": asdict(h["train"]),
            "val": asdict(h["val"]),
        }
        for h in history
    ]


def run_training(cfg: Config, data: EpochedData, output_dir: Union[str, Path]) -> dict:
    """Train, evaluate, and write all artifacts under ``output_dir``.

    Returns a dict with the epoch ``history``, the ``val`` and ``test`` results,
    and a JSON-friendly ``summary``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.train.seed)

    loaders = build_dataloaders(data, cfg)
    trainer = Trainer(build_model(cfg), cfg)
    history = trainer.fit(
        loaders.train, loaders.val, checkpoint_path=output_dir / "best_model.pt"
    )

    val_result = trainer.evaluate(loaders.val)
    y_true, y_pred = trainer.predict(loaders.test)
    test_metrics = compute_metrics(y_true, y_pred, cfg.model.num_classes)

    cfg.to_yaml(output_dir / "config.yaml")
    plot_history(history, output_dir / "training_curves.png")
    plot_confusion(
        test_metrics.confusion,
        _class_names(cfg.model.num_classes),
        output_dir / "confusion_matrix.png",
    )

    summary = {
        "val_accuracy": val_result.accuracy,
        "test_accuracy": test_metrics.accuracy,
        "test_macro_f1": test_metrics.macro_f1,
        "test_kappa": test_metrics.kappa,
        "confusion": test_metrics.confusion.tolist(),
        "epochs_run": len(history),
        "split_sizes": {
            "train": int(len(loaders.split.train)),
            "val": int(len(loaders.split.val)),
            "test": int(len(loaders.split.test)),
        },
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"summary": summary, "history": _history_to_jsonable(history)}, fh, indent=2
        )

    return {
        "history": history,
        "val": val_result,
        "test": test_metrics,
        "summary": summary,
    }
