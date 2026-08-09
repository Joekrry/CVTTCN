"""Plotting helpers for training curves and the confusion matrix.

Uses the non-interactive ``Agg`` backend so figures can be written to disk on a
headless machine without opening a window.
"""

from pathlib import Path
from typing import Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import numpy as np


def _ensure_parent(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_history(history: Sequence[dict], path: Union[str, Path]) -> None:
    """Plot train/val loss and accuracy over epochs to ``path``."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [h["train"].loss for h in history], label="train")
    ax_loss.plot(epochs, [h["val"].loss for h in history], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, [h["train"].accuracy for h in history], label="train")
    ax_acc.plot(epochs, [h["val"].accuracy for h in history], label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(_ensure_parent(path), dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], path: Union[str, Path]) -> None:
    """Plot a confusion matrix with counts annotated to ``path``."""
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(_ensure_parent(path), dpi=120)
    plt.close(fig)
