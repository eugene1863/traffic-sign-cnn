"""Evaluates a trained model on the official GTSRB held-out test set
(data/Test + data/Test.csv) and writes a classification report, confusion
matrix, and a grid of misclassified examples to outputs/.

Usage:
    python -m src.evaluate
    python -m src.evaluate --model models/traffic_sign_cnn_final.keras
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from . import config
from .class_names import CLASS_NAMES, NUM_CLASSES
from .data import load_official_test_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.BEST_MODEL_PATH)
    p.add_argument("--test-dir", default=config.TEST_DIR)
    p.add_argument("--test-csv", default=config.TEST_CSV)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model = keras.models.load_model(args.model)

    print(f"Loading official test set from {args.test_csv} ...")
    images, labels = load_official_test_set(args.test_dir, args.test_csv, args.img_size)
    print(f"{len(images)} test images loaded.")

    probs = model.predict(images, batch_size=128, verbose=1)
    preds = np.argmax(probs, axis=1)

    acc = float(np.mean(preds == labels))
    print(f"\nTest accuracy: {acc:.4f}\n")

    report = classification_report(
        labels,
        preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    )
    print(report)
    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, ax=ax, cmap="Blues", square=True, cbar=True)
    ax.set_xlabel("Predicted class id")
    ax.set_ylabel("True class id")
    ax.set_title(f"Confusion matrix (test accuracy {acc:.3f})")
    fig.tight_layout()
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    misclassified = np.where(preds != labels)[0]
    if len(misclassified) > 0:
        n_show = min(16, len(misclassified))
        chosen = np.random.default_rng(config.SEED).choice(misclassified, n_show, replace=False)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for ax, idx in zip(axes.flat, chosen):
            ax.imshow(images[idx].astype("uint8"))
            true_name = CLASS_NAMES[labels[idx]]
            pred_name = CLASS_NAMES[preds[idx]]
            conf = probs[idx, preds[idx]]
            ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.2f})", fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(chosen):]:
            ax.axis("off")
        fig.tight_layout()
        mis_path = os.path.join(config.OUTPUTS_DIR, "misclassified_examples.png")
        fig.savefig(mis_path, dpi=150)
        plt.close(fig)
        print(f"Saved {n_show} misclassified examples to {mis_path}")


if __name__ == "__main__":
    main()
