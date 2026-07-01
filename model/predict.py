"""
Kidney Disease Detection — Single Image Prediction (PyTorch)
============================================================
Usage:
  python predict.py --image path/to/ct_scan.jpg
  python predict.py --image path/to/ct_scan.jpg --json
"""

import os
import sys
import argparse
import json
import numpy as np

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import CLASS_NAMES, IMAGE_SIZE, load_model, get_device, IMAGENET_MEAN, IMAGENET_STD

# ─── Class metadata ────────────────────────────────────────────────────────────
CLASS_DESCRIPTIONS = {
    "Normal": (
        "No kidney abnormalities detected. The kidney tissue appears healthy "
        "with normal structure and density."
    ),
    "Cyst": (
        "A fluid-filled sac (cyst) detected in the kidney. Most kidney cysts "
        "are benign but should be monitored by a specialist."
    ),
    "Tumor": (
        "Abnormal tissue growth (tumor) detected. Immediate consultation with "
        "a urologist or oncologist is strongly recommended."
    ),
    "Stone": (
        "Calcification (kidney stone) detected. Stones may cause pain and "
        "urinary complications. Consult a urologist for treatment options."
    ),
}

SEVERITY = {
    "Normal": "✅ Low Risk",
    "Cyst":   "⚠️  Moderate Risk",
    "Tumor":  "🔴 High Risk",
    "Stone":  "⚠️  Moderate Risk",
}


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load, resize and normalise an image to (1, 3, 224, 224) tensor."""
    try:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        img = Image.open(image_path).convert("RGB")
        return tf(img).unsqueeze(0)  # (1, 3, 224, 224)
    except ImportError:
        # Fallback: manual numpy preprocessing without torchvision
        img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE[::-1], Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        arr  = (arr - mean) / std
        arr  = arr.transpose(2, 0, 1)           # HWC → CHW
        return torch.from_numpy(arr).unsqueeze(0)  # (1, 3, 224, 224)


def predict(image_path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Run inference and return structured result dict."""
    x     = preprocess_image(image_path).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    idx       = int(np.argmax(probs))
    cls_name  = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return {
        "predicted_class": cls_name,
        "confidence":      round(confidence, 4),
        "probabilities":   {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        "description":     CLASS_DESCRIPTIONS[cls_name],
        "severity":        SEVERITY[cls_name],
    }


def main():
    parser = argparse.ArgumentParser(description="Kidney Disease Detection — Predict")
    parser.add_argument("--image", required=True, type=str, help="Path to CT scan image")
    parser.add_argument("--model", type=str,
                        default=os.path.join(os.path.dirname(__file__), "kidney_model.pth"),
                        help="Path to .pth model weights (default: kidney_model.pth)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌  Image not found: {args.image}"); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"❌  Model not found: {args.model}")
        print("    Train first: python train.py --demo"); sys.exit(1)

    device = get_device()
    model  = load_model(args.model, str(device)).to(device)
    result = predict(args.image, model, device)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 55)
        print("  Kidney Disease Detection — Prediction Result")
        print("=" * 55)
        print(f"  Image           : {args.image}")
        print(f"  Predicted Class : {result['predicted_class']}")
        print(f"  Confidence      : {result['confidence']*100:.2f}%")
        print(f"  Risk Level      : {result['severity']}")
        print(f"\n  Description:\n  {result['description']}")
        print("\n  Class Probabilities:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {cls:<10} {bar:<30} {prob*100:5.1f}%")
        print("=" * 55)
        print("  ⚕  Research purposes only. Consult a medical professional.")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
