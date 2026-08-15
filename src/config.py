"""Central configuration. Override any of these from the CLI flags exposed
by train.py / evaluate.py / predict.py instead of editing this file, unless
you're changing a project-wide default.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")          # data/Train/<class_id>/*.png
TEST_DIR = os.path.join(DATA_DIR, "Test")            # data/Test/*.png
TEST_CSV = os.path.join(DATA_DIR, "Test.csv")        # official GTSRB test labels
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

IMG_SIZE = 48          # signs are resized to IMG_SIZE x IMG_SIZE
CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 40
VAL_SPLIT = 0.2
SEED = 42
LEARNING_RATE = 1e-3

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "traffic_sign_cnn_best.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "traffic_sign_cnn_final.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "traffic_sign_cnn.tflite")
