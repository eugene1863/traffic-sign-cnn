"""Data loading and augmentation for the GTSRB traffic-sign dataset.

Expected layout (this is the standard Kaggle "GTSRB - German Traffic Sign
Recognition Benchmark" layout, see README.md for the download step):

    data/
      Train/
        0/   *.png
        1/   *.png
        ...
        42/  *.png
      Test/
        *.png
      Test.csv        # columns include Path,ClassId for the official test set

`Train/` is split into train/validation here. `Test/` + `Test.csv` is the
held-out official benchmark set, only touched by evaluate.py.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config


def _numeric_class_names(directory: str) -> list[str]:
    """Folder names are '0'..'42'. Sorting them as plain strings gives
    '0','1','10','11',... which would scramble label indices relative to
    class_names.CLASS_NAMES. Force numeric order instead.
    """
    present = {d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))}
    ordered = [str(i) for i in range(43) if str(i) in present]
    return ordered


def make_train_val_datasets(
    train_dir: str = config.TRAIN_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.SEED,
):
    """Builds train/validation tf.data.Dataset pipelines from data/Train."""
    class_names = _numeric_class_names(train_dir)

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds


def load_official_test_set(
    test_dir: str = config.TEST_DIR,
    test_csv: str = config.TEST_CSV,
    img_size: int = config.IMG_SIZE,
):
    """Loads the official GTSRB Test.csv holdout set as numpy arrays.

    Returns (images, labels) with images scaled to [0, 255] uint8-range
    float32 (matching the Rescaling layer inside the model), shape
    (N, img_size, img_size, 3).
    """
    df = pd.read_csv(test_csv)
    data_root = os.path.dirname(test_csv)

    images = np.zeros((len(df), img_size, img_size, 3), dtype=np.float32)
    labels = np.zeros((len(df),), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        path = os.path.join(data_root, row.Path)
        img = keras.utils.load_img(path, target_size=(img_size, img_size))
        images[i] = keras.utils.img_to_array(img)
        labels[i] = int(row.ClassId)

    return images, labels


def class_weights_from_directory(train_dir: str = config.TRAIN_DIR) -> dict:
    """Traffic-sign classes are naturally imbalanced (e.g. many more
    'Speed limit 50' examples than 'Go straight or left'). Compute
    inverse-frequency class weights so training doesn't just learn to
    predict the majority classes.
    """
    class_names = _numeric_class_names(train_dir)
    counts = {}
    for cname in class_names:
        folder = os.path.join(train_dir, cname)
        counts[int(cname)] = len(os.listdir(folder))

    total = sum(counts.values())
    n_classes = len(counts)
    return {cid: total / (n_classes * count) for cid, count in counts.items()}
