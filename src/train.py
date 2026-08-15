"""Trains the traffic-sign CNN on data/Train (see README.md for how to get
the data) and saves the best checkpoint plus training curves.

Usage:
    python -m src.train
    python -m src.train --epochs 20 --batch-size 32
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a GUI window
import matplotlib.pyplot as plt
from tensorflow import keras

from . import config
from .class_names import NUM_CLASSES
from .data import class_weights_from_directory, make_train_val_datasets
from .model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", default=config.TRAIN_DIR)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--no-class-weights", action="store_true", help="disable inverse-frequency class weighting")
    return p.parse_args()


def plot_history(history: keras.callbacks.History, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print(f"Loading data from {args.train_dir} ...")
    train_ds, val_ds = make_train_val_datasets(
        train_dir=args.train_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    class_weight = None
    if not args.no_class_weights:
        class_weight = class_weights_from_directory(args.train_dir)
        print(f"Using inverse-frequency class weights for {len(class_weight)} classes.")

    model = build_model(num_classes=NUM_CLASSES, img_size=args.img_size, learning_rate=args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(config.FINAL_MODEL_PATH)
    print(f"Saved final model to {config.FINAL_MODEL_PATH}")
    print(f"Best checkpoint (by val_accuracy) saved to {config.BEST_MODEL_PATH}")

    history_path = os.path.join(config.OUTPUTS_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    plot_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
    plot_history(history, plot_path)
    print(f"Saved training curves to {plot_path}")


if __name__ == "__main__":
    main()
