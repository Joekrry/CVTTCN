# CVTTCN

A hybrid Convolutional Vision Transformer (CvT) + Temporal Convolutional
Network (TCN), implemented in PyTorch, for EEG motor-imagery classification
on the EEGMMIDB (PhysioNet EEG Motor Movement/Imagery) dataset.

Task: binary classification of imagined left-fist vs right-fist movement
(runs 4, 8, 12), pooled across all subjects.

## Architecture

```
input (B, 1, 64 channels, T)
  -> Conv token embedding        (collapses channels, tokenizes time)
  -> CvT transformer blocks      (conv-projected Q/K/V, no positional embedding)
  -> transpose to (B, d_model, T')
  -> TCN                         (dilated causal convolutions)
  -> global average pool -> Linear(num_classes)
```

## Setup

Requires Python 3.11+ (developed on 3.14). Everything runs inside a
project-local virtual environment.

```bash
python -m venv .venv
.venv/Scripts/activate   # on Windows; use .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA build of PyTorch by default. Edit the
torch line if you are on a CPU-only machine; the code picks CPU automatically
when CUDA is unavailable (`device: auto` in the config).

## Usage

Train:

```bash
python scripts/train.py --config configs/default.yaml --cache data/epochs.npz
```

The first run downloads and preprocesses the dataset via MNE and caches it
to `--cache`; later runs reuse the cache. Artifacts (best checkpoint, config,
training curves, confusion matrix, `results.json`) are written to
`--output-dir` (default `results/`).

Evaluate a saved checkpoint:

```bash
python scripts/evaluate.py --checkpoint results/best_model.pt --config configs/default.yaml
```

Run the tests:

```bash
pytest
```

## Configs

- `configs/default.yaml` -- the rigorous, no-leakage protocol: whole epochs,
  pooled cross-subject split performed at *trial* granularity so no trial's
  data crosses the train/val/test boundary.
- `configs/windowed.yaml` -- crops each epoch into overlapping windows and
  splits at the window level, the protocol commonly used in the EEGMMIDB
  literature. Scores higher because overlapping windows from the same trial
  can end up on both sides of the split.

## Results

| config | protocol | test accuracy |
|---|---|---|
| `default.yaml` | trial-level split, no leakage | ~0.67 |
| `windowed.yaml` | window-level split (literature protocol) | ~0.82 |

The trial-level number is the honest measure of cross-subject generalization
on this dataset: EEGMMIDB includes subjects with little separable motor-imagery
signal ("BCI illiteracy"), which caps pooled cross-subject accuracy well below
within-subject or window-leaked numbers. The windowed number reproduces the
80-90% range commonly reported for this architecture class and confirms the
model itself is not the bottleneck.

## Project layout

```
configs/            YAML configs (default, windowed)
scripts/             train.py, evaluate.py
src/cvttcn/
  config.py          typed config schema
  data/              download, preprocessing, dataset, splitting
  models/            CvT blocks, TCN, hybrid model
  training/          metrics, trainer, seed/device utils
  experiment.py       end-to-end run orchestration
  plots.py           training-curve and confusion-matrix plots
tests/               pytest suite
```
