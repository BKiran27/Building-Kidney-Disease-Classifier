"""
Kidney Disease Detection — Flask REST API (PyTorch)
====================================================
Endpoints:
  GET  /              → Serve web UI
  POST /api/predict   → Accept image, return prediction JSON
  GET  /api/health    → Health check
  GET  /api/model-info → Model metadata

Usage:
  python app.py
  Open http://localhost:5000
"""

import os
import sys
import io
import json
import traceback
import random
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch

# ─── Path setup ───────────────────────────────────────────────────────────────
API_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(API_DIR)
MODEL_DIR  = os.path.join(ROOT_DIR, "model")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
MODEL_PATH = os.path.join(MODEL_DIR, "kidney_model.pth")

sys.path.insert(0, MODEL_DIR)
from model   import CLASS_NAMES, IMAGE_SIZE, load_model, get_device, IMAGENET_MEAN, IMAGENET_STD
from predict import CLASS_DESCRIPTIONS, SEVERITY, preprocess_image

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

# ─── Model (lazy-loaded once) ─────────────────────────────────────────────────
_model  = None
_device = get_device()


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH, str(_device)).to(_device)
        print(f"✅  Model loaded from {MODEL_PATH}  (device={_device})")
    return _model


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    model = get_model()
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "device":       str(_device),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "classes":      CLASS_NAMES,
    })


@app.route("/api/model-info")
def model_info():
    model = get_model()
    info = {
        "model_name":   "KidneyDiseaseNet",
        "architecture": "ResNet50 + Custom Head (PyTorch)",
        "classes":      CLASS_NAMES,
        "input_size":   f"{IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}",
        "device":       str(_device),
        "loaded":       model is not None,
    }
    if model is not None:
        total = sum(p.numel() for p in model.parameters())
        info["total_params"] = total
    return jsonify(info)


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """Accept image upload, return prediction JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Use field name 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed:
        return jsonify({"error": f"Unsupported type '{ext}'.  Allowed: {', '.join(allowed)}"}), 400

    model = get_model()

    # ── Demo mode (model not trained) ────────────────────────────────────────
    if model is None:
        random.seed(hash(file.filename) % 9999)
        probs   = np.abs(np.random.dirichlet(np.ones(len(CLASS_NAMES)))).tolist()
        idx     = int(np.argmax(probs))
        cls     = CLASS_NAMES[idx]
        return jsonify({
            "predicted_class": cls,
            "confidence":      round(probs[idx], 4),
            "probabilities":   {n: round(p, 4) for n, p in zip(CLASS_NAMES, probs)},
            "description":     CLASS_DESCRIPTIONS[cls],
            "severity":        SEVERITY[cls],
            "demo_mode":       True,
            "message":         "Model not trained yet — showing demo prediction.",
        })

    # ── Real inference ────────────────────────────────────────────────────────
    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img       = img.resize(IMAGE_SIZE[::-1], Image.LANCZOS)

        arr  = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        arr  = (arr - mean) / std
        arr  = arr.transpose(2, 0, 1)  # HWC → CHW
        x    = torch.from_numpy(arr).unsqueeze(0).to(_device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        cls = CLASS_NAMES[idx]

        return jsonify({
            "predicted_class": cls,
            "confidence":      round(float(probs[idx]), 4),
            "probabilities":   {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, probs)},
            "description":     CLASS_DESCRIPTIONS[cls],
            "severity":        SEVERITY[cls],
            "demo_mode":       False,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 55)
    print("  KidneyScan AI -- Flask API (PyTorch)")
    print("=" * 55)
    get_model()
    print("\n[*] Server running at http://localhost:5000")
    print("    Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
