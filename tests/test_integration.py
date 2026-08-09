"""Fast end-to-end integration smoke test.

Runs the whole training orchestration on tiny synthetic data (CPU, no network,
no download) and checks that it learns a trivial signal and writes every
artifact. Keeps the real >70% run out of the test suite while still exercising
the full path.
"""

import numpy as np

from cvttcn.config import Config
from cvttcn.data.preprocessing import EpochedData
from cvttcn.experiment import run_training
from cvttcn.training.utils import set_seed


def _toy_epoched(n_per_class=40, n_ch=8, n_time=64, seed=0) -> EpochedData:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.1, (n_per_class, 1, n_ch, n_time))
    x1 = rng.normal(1.0, 0.1, (n_per_class, 1, n_ch, n_time))
    X = np.concatenate([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    n = len(y)
    subjects = np.zeros(n, dtype=np.int64)
    trials = np.arange(n, dtype=np.int64)
    return EpochedData(X, y, subjects, trials)


def _toy_config() -> Config:
    return Config.from_dict(
        {
            "data": {"n_channels": 8},
            "model": {
                "cvt": {"embed_dim": 16, "depth": 1, "num_heads": 2},
                "tcn": {"channels": [16]},
            },
            "train": {
                "epochs": 3,
                "batch_size": 16,
                "device": "cpu",
                "amp": False,
                "warmup_epochs": 1,
                "early_stopping_patience": 99,
            },
        }
    )


def test_run_training_writes_artifacts_and_returns_metrics(tmp_path):
    result = run_training(_toy_config(), _toy_epoched(), tmp_path)

    assert len(result["history"]) >= 1
    assert result["test"].confusion.shape == (2, 2)
    assert 0.0 <= result["summary"]["test_accuracy"] <= 1.0

    for artifact in (
        "best_model.pt",
        "config.yaml",
        "training_curves.png",
        "confusion_matrix.png",
        "results.json",
    ):
        assert (tmp_path / artifact).exists(), f"missing artifact: {artifact}"


def test_run_training_learns_toy_signal(tmp_path):
    result = run_training(_toy_config(), _toy_epoched(), tmp_path)
    # A trivially separable task should be classified well on the held-out test.
    assert result["summary"]["test_accuracy"] > 0.8
