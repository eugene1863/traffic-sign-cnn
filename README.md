# Traffic Sign Recognition CNN

A CNN that classifies traffic signs into the 43 categories of the GTSRB
(German Traffic Sign Recognition Benchmark) dataset, meant as a building
block for a driver-assistance feature (e.g. "warn the driver what sign
was just seen" after a detector has already cropped it out of a dash-cam
frame).

**Scope note:** this is a *classifier*, not a *detector*. It expects an
image that is already roughly cropped to one sign (which is how GTSRB is
built, and how a real pipeline would feed it after a detection stage like
YOLO). It does not find signs in a full road scene by itself. It is a
prototype/learning project, not a safety-certified system — don't wire
its output directly into vehicle controls.

## Project layout

```
src/
  class_names.py   43 GTSRB class labels (index == class id)
  config.py        paths, image size, hyperparameters
  data.py          dataset loading, class-imbalance weighting
  model.py         CNN architecture (augmentation baked in)
  train.py         training loop, checkpoints, curves
  evaluate.py      official test-set metrics, confusion matrix
  predict.py       single/batch inference, TFLite export
download_data.py   fetches GTSRB from Kaggle
data/              Train/, Test/, Test.csv (not included, see below)
models/            saved .keras / .tflite checkpoints (created by train.py)
outputs/           plots, reports (created by train.py / evaluate.py)
```

## Setup

```bash
pip install -r requirements.txt
```

## Getting the data

This project uses the "GTSRB - German Traffic Sign Recognition Benchmark"
dataset (43 classes, ~50k images), via Kaggle:

```bash
python download_data.py
```

That needs a free Kaggle account + API token (`~/.kaggle/kaggle.json`) —
the script prints exact steps if it can't find one. Alternatively, download
manually from
[kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
and unzip so you end up with:

```
data/Train/0/...png  ... data/Train/42/...png
data/Test/...png
data/Test.csv
```

Any other folder-per-class dataset (`<root>/<class_name>/*.jpg`) also works
for training — swap in your own directory with `--train-dir`. `evaluate.py`
specifically needs GTSRB's `Test.csv` format for its held-out benchmark.

## Train

```bash
python -m src.train                          # defaults: 40 epochs, batch 64, 48x48 input
python -m src.train --epochs 20 --batch-size 32 --lr 5e-4
```

Saves `models/traffic_sign_cnn_best.keras` (best validation accuracy,
via `ModelCheckpoint`) and `models/traffic_sign_cnn_final.keras`, plus
`outputs/training_curves.png` and `outputs/history.json`. Training uses
early stopping and LR reduction on plateau, and inverse-frequency class
weights (GTSRB classes are naturally imbalanced — some signs have 10x more
training examples than others).

## Evaluate

```bash
python -m src.evaluate
```

Runs the saved model against GTSRB's official `Test.csv` holdout and writes
to `outputs/`: `classification_report.txt` (per-class precision/recall/F1),
`confusion_matrix.png`, and `misclassified_examples.png` (a grid of the
model's actual mistakes, useful for spotting which signs get confused —
e.g. speed limit digits are a classic failure mode).

## Predict

```bash
python -m src.predict --image path/to/sign.jpg
python -m src.predict --image-dir path/to/folder
python -m src.predict --export-tflite     # for embedded/mobile deployment
```

Single/batch inference prints the top-k predictions and saves an annotated
copy of each image with the predicted label overlaid. Predictions below
60% confidence are flagged `LOW CONFIDENCE` rather than treated as reliable
— for a driver-facing tool, a confident wrong answer is worse than an
honest "not sure."

## Model

Three convolutional blocks (32 → 64 → 128 filters, each Conv-BN-Conv-BN-
MaxPool-Dropout) feeding a 256-unit dense head, ~1.5M parameters. Input
resizing/rescaling and augmentation (rotation, zoom, translation,
contrast — **no flips**, since flipping a directional sign like "turn
left" would silently change its meaning) are Keras layers baked into the
model graph, so `predict.py` needs no separate preprocessing code and
augmentation automatically turns itself off outside of `model.fit`.

On the real GTSRB dataset this architecture typically reaches ~98%+ test
accuracy after full training; expect a few epochs' warmup before validation
accuracy climbs, since the early layers are learning basic edge/color
features from scratch.
