"""
config.py — Centralised configuration for KidneyScan AI
=========================================================
All hyperparameters and paths live here so scripts are
free of magic numbers.
"""

import os

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(ROOT_DIR, "model")
API_DIR    = os.path.join(ROOT_DIR, "api")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(MODEL_DIR, "outputs")

# ─── Saved Model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "kidney_model.pth")

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Default Kaggle dataset folder name
DEFAULT_DATASET = os.path.join(
    DATA_DIR,
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    "Normal": (
        "No kidney abnormalities detected. "
        "The kidney tissue appears healthy with normal structure and density."
    ),
    "Cyst": (
        "A fluid-filled sac (cyst) detected in the kidney. "
        "Most kidney cysts are benign but should be monitored by a specialist."
    ),
    "Tumor": (
        "Abnormal tissue growth (tumor) detected. "
        "Immediate consultation with a urologist or oncologist is strongly recommended."
    ),
    "Stone": (
        "Calcification (kidney stone) detected. "
        "Stones may cause pain and urinary complications. "
        "Consult a urologist for treatment options."
    ),
}

SEVERITY = {
    "Normal": "Low Risk",
    "Cyst":   "Moderate Risk",
    "Tumor":  "High Risk",
    "Stone":  "Moderate Risk",
}

CLASS_DESCRIPTIONS = {
    "Normal": (
        "No kidney abnormalities detected. "
        "The kidney tissue appears healthy with normal structure and density."
    ),
    "Cyst": (
        "A fluid-filled sac (cyst) detected in the kidney. "
        "Most kidney cysts are benign but should be monitored by a specialist."
    ),
    "Tumor": (
        "Abnormal tissue growth (tumor) detected. "
        "Immediate consultation with a urologist or oncologist is strongly recommended."
    ),
    "Stone": (
        "Calcification (kidney stone) detected. "
        "Stones may cause pain and urinary complications. "
        "Consult a urologist for treatment options."
    ),
}

SEVERITY = {
    "Normal": "Low Risk",
    "Cyst":   "Moderate Risk",
    "Tumor":  "High Risk",
    "Stone":  "Moderate Risk",
}

# ─── Image ────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
INPUT_SHAPE  = (3, 224, 224)   # PyTorch CHW format

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 30
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-4
VAL_SPLIT         = 0.20     # 80/20 train–val split
SEED              = 42
UNFREEZE_TOP_N    = 2        # Number of ResNet50 blocks to fine-tune

# LR Scheduler
LR_FACTOR   = 0.5
LR_PATIENCE = 4
LR_MIN      = 1e-7

# Early stopping (real data mode only)
ES_PATIENCE = 8

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 64
