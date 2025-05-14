# src/config.py

import os
from pathlib import Path

# === Пути к данным ===
# BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR    = Path("/content/drive/MyDrive/FakeNewNet")
DATA_DIR    = BASE_DIR / "data"
TRAIN_PATH  = DATA_DIR / "train.csv"
VAL_PATH    = DATA_DIR / "val.csv"
TEST_PATH   = DATA_DIR / "test.csv"

# === Пути к моделям и логам ===
MODEL_DIR   = BASE_DIR / "src/models"
LOG_DIR     = BASE_DIR / "src/logs"
OUTPUT_DIR  = BASE_DIR / "src/outputs"

# === Гиперпараметры ===
MAX_LEN         = 256
BATCH_SIZE      = 16
EPOCHS          = 3
LEARNING_RATE   = 2e-5
SEED            = 42

# === HuggingFace модель ===
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

# === Классы ===
LABELS = ["real", "fake"]

# === Использование GPU ===
DEVICE = "cuda"

FREEZE_LAYERS = 3

# === Размер вектора ручных (hand-crafted) признаков ===
HAND_FDIM = 12
