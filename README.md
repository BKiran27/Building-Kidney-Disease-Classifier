<div align="center">

# 🩺 KidneyScan AI
### Deep Learning Kidney Disease Detection from CT Scans

[![CI](https://github.com/BKiran27/Building-Kidney-Disease-Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/BKiran27/Building-Kidney-Disease-Classifier/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A full-stack AI system that classifies kidney CT scan images into four disease categories using ResNet50 transfer learning — served through a beautiful dark-mode web application.**

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#-architecture) · [📊 Results](#-results) · [🌐 API Reference](#-api-reference) · [🐳 Docker](#-docker)

</div>

---

## 🔬 What It Detects

| Class | Description | Risk Level |
|-------|-------------|------------|
| ✅ **Normal** | Healthy kidney tissue, no abnormalities | Low |
| 🔵 **Cyst** | Fluid-filled sac — usually benign, monitor regularly | Moderate |
| 🟠 **Stone** | Calcified mineral deposits causing obstruction | Moderate |
| 🔴 **Tumor** | Abnormal tissue growth — immediate specialist evaluation | High |

---

## ✨ Features

- **ResNet50 Transfer Learning** — 24.7M parameter model fine-tuned on kidney CT scans
- **Test-Time Augmentation (TTA)** — improved prediction accuracy via multi-view inference
- **Premium Dark-Mode Web UI** — drag-and-drop upload, animated confidence ring, probability bars
- **Flask REST API** — `/api/predict`, `/api/health`, `/api/model-info`
- **Demo Mode** — runs without a trained model (random predictions for UI testing)
- **Full Evaluation Suite** — confusion matrix, ROC-AUC curves, per-class accuracy
- **Docker Support** — single-command deployment
- **GitHub Actions CI** — automatic linting, unit tests, and model sanity checks

---

## 🏗️ Architecture

```
Input CT Scan (224×224×3)
         │
         ▼
┌─────────────────────┐
│   ResNet50 Backbone │  ← ImageNet pretrained (top 2 blocks fine-tuned)
│   (23.5M params)    │
└─────────────────────┘
         │ GlobalAvgPool → 2048D vector
         ▼
┌─────────────────────┐
│  Dense(512) + BN    │
│  + ReLU + Drop(0.4) │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Dense(256) + BN    │
│  + ReLU + Drop(0.3) │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dense(4, Softmax) │  → [Cyst | Normal | Stone | Tumor]
└─────────────────────┘
```

**Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
**Loss:** Cross-Entropy
**Scheduler:** ReduceLROnPlateau (factor=0.5, patience=4)

---

## 📁 Project Structure

```
Building-Kidney-Disease-Classifier/
│
├── model/
│   ├── model.py         # KidneyDiseaseNet — ResNet50 + custom head
│   ├── train.py         # Training pipeline (demo + real dataset)
│   ├── evaluate.py      # Evaluation metrics & plots
│   ├── predict.py       # Single-image CLI inference
│   └── outputs/         # Generated plots after training
│       ├── training_history.png
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       └── per_class_accuracy.png
│
├── api/
│   ├── app.py           # Flask REST API
│   ├── requirements.txt # Production deps
│   └── requirements-dev.txt  # Dev/test deps
│
├── static/
│   ├── index.html       # Premium web UI
│   ├── style.css        # Glassmorphism dark-mode styles
│   └── app.js           # Drag-drop, fetch, animated results
│
├── tests/
│   ├── test_model.py    # Model architecture unit tests
│   └── test_api.py      # Flask API endpoint tests
│
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI pipeline
│
├── config.py            # Centralised configuration
├── utils.py             # Shared utilities (transforms, plotting, TTA)
├── Dockerfile           # Container image
├── docker-compose.yml   # Docker Compose config
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (3.11 recommended for Docker)
- pip or conda

### 1 — Clone & Install

```bash
git clone https://github.com/BKiran27/Building-Kidney-Disease-Classifier.git
cd Building-Kidney-Disease-Classifier

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
pip install -r api/requirements.txt
```

### 2 — Launch the Web App

```bash
python api/app.py
```

Open **http://localhost:5000** in your browser.

> The app immediately runs in **Demo Mode** — upload any image and see the full UI in action. Real predictions require training the model first.

---

## 🤖 Training

### Option A — Demo Mode *(no dataset, ~2 min)*

```bash
python model/train.py --demo
```

Trains on synthetic data for 10 epochs to verify the pipeline end-to-end.

### Option B — Real Dataset *(Kaggle CT-KIDNEY)*

**Step 1:** Download the dataset from Kaggle:
> [CT KIDNEY DATASET: Normal-Cyst-Tumor-Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

**Step 2:** Extract so the structure looks like:

```
data/
└── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
    ├── Cyst/
    ├── Normal/
    ├── Stone/
    └── Tumor/
```

**Step 3:** Train:

```bash
python model/train.py \
  --data ./data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \
  --epochs 30 \
  --batch 32 \
  --lr 1e-4
```

Training outputs:
- `model/kidney_model.pth` — best model weights
- `model/outputs/training_history.png` — accuracy & loss curves

---

## 📊 Results

After training, run evaluation:

```bash
# Demo evaluation
python model/evaluate.py --demo

# Real dataset evaluation
python model/evaluate.py \
  --data ./data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
```

**Generated Outputs:**

| Plot | Description |
|------|-------------|
| `confusion_matrix.png` | Actual vs predicted class heatmap |
| `roc_curves.png` | One-vs-Rest ROC-AUC per class |
| `per_class_accuracy.png` | Bar chart of per-class accuracy |
| `classification_report.txt` | Precision, Recall, F1 per class |
| `metrics.json` | Overall accuracy and macro-AUC |

---

## 🔍 Single Image CLI

```bash
# Human-readable output
python model/predict.py --image path/to/ct_scan.jpg

# JSON output (for scripting)
python model/predict.py --image path/to/ct_scan.jpg --json
```

**Example JSON output:**
```json
{
  "predicted_class": "Cyst",
  "confidence": 0.9214,
  "probabilities": {
    "Cyst":   0.9214,
    "Normal": 0.0512,
    "Stone":  0.0198,
    "Tumor":  0.0076
  },
  "description": "A fluid-filled sac (cyst) detected...",
  "severity": "Moderate Risk"
}
```

---

## 🌐 API Reference

Base URL: `http://localhost:5000`

### `POST /api/predict`

Upload a CT scan image for classification.

**Request:** `multipart/form-data`, field: `image`

**Response:**
```json
{
  "predicted_class": "Stone",
  "confidence": 0.8833,
  "probabilities": { "Cyst": 0.04, "Normal": 0.02, "Stone": 0.88, "Tumor": 0.06 },
  "description": "Calcification (kidney stone) detected...",
  "severity": "Moderate Risk",
  "demo_mode": false
}
```

### `GET /api/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "classes": ["Cyst", "Normal", "Stone", "Tumor"],
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### `GET /api/model-info`

```json
{
  "model_name": "KidneyDiseaseNet",
  "architecture": "ResNet50 + Custom Head (PyTorch)",
  "classes": ["Cyst", "Normal", "Stone", "Tumor"],
  "input_size": "224×224",
  "total_params": 24691012,
  "loaded": true
}
```

---

## 🐳 Docker

### Build & Run

```bash
# Build image
docker build -t kidneyai .

# Run container
docker run -p 5000:5000 kidneyai
```

### Docker Compose

```bash
docker-compose up -d
```

App available at **http://localhost:5000**

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install -r api/requirements-dev.txt

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=html
```

---

## 🛠️ Development

```bash
# Format code
black . --line-length 120

# Sort imports
isort .

# Lint
flake8 . --max-line-length 120
```

---

## ⚕️ Medical Disclaimer

> **This tool is for research and educational purposes only.**
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.
> **Always consult a qualified healthcare provider** for any medical concerns.
> The AI predictions may be inaccurate and should never be used as the sole basis for clinical decisions.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [CT KIDNEY DATASET — Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/) teams
- [ResNet50](https://arxiv.org/abs/1512.03385) — He et al., 2015

---

<div align="center">
Made with ❤️ by <a href="https://github.com/BKiran27">BKiran27</a>
</div>
