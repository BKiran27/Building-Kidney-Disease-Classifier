"""
predict.py — KidneyScan AI  (v2 with TTA)
==========================================
Usage:
  python model/predict.py --image ct_scan.jpg
  python model/predict.py --image ct_scan.jpg --tta --json
"""

import os
import sys
import argparse
import json
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MODEL_PATH, CLASS_DESCRIPTIONS, SEVERITY,
)
from model.model import load_model, get_device

try:
    from torchvision import transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False


def get_val_tf():
    if HAS_TV:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return None


def get_tta_tf():
    if not HAS_TV:
        return []
    return [
        transforms.Compose([
            transforms.Resize((int(IMAGE_SIZE[0]*1.1), int(IMAGE_SIZE[1]*1.1))),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(10),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((int(IMAGE_SIZE[0]*1.1), int(IMAGE_SIZE[1]*1.1))),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]


def preprocess_image(image_source) -> torch.Tensor:
    """Return (1, 3, H, W) tensor — no TTA."""
    img = _load_pil(image_source)
    tf  = get_val_tf()
    if tf:
        return tf(img).unsqueeze(0)
    # Fallback: manual numpy
    img_r = img.resize(IMAGE_SIZE[::-1], Image.LANCZOS)
    arr   = np.array(img_r, dtype=np.float32) / 255.0
    mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
    std   = np.array(IMAGENET_STD,  dtype=np.float32)
    arr   = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _load_pil(source) -> Image.Image:
    if isinstance(source, (str, os.PathLike)):
        return Image.open(source).convert("RGB")
    if isinstance(source, bytes):
        import io
        return Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    raise TypeError(f"Unsupported type: {type(source)}")


def predict(image_source, model: torch.nn.Module, device: torch.device,
            use_tta: bool = False) -> dict:
    """Run inference and return result dict."""
    img  = _load_pil(image_source)
    tf   = get_val_tf()

    # Build batch: original + TTA augments
    if use_tta and tf is not None:
        views = [tf(img).unsqueeze(0)] + [t(img).unsqueeze(0) for t in get_tta_tf()]
        x     = torch.cat(views, dim=0).to(device)   # (N_aug, 3, H, W)
    else:
        x = preprocess_image(img).to(device)          # (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        if use_tta and probs.shape[0] > 1:
            probs = probs.mean(dim=0, keepdim=True)   # average TTA predictions
        probs = probs[0].cpu().numpy()

    idx        = int(np.argmax(probs))
    cls_name   = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return {
        "predicted_class": cls_name,
        "confidence":      round(confidence, 4),
        "probabilities":   {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, probs)},
        "description":     CLASS_DESCRIPTIONS[cls_name],
        "severity":        SEVERITY[cls_name],
        "tta_used":        use_tta,
    }


def main():
    parser = argparse.ArgumentParser(description="KidneyScan AI — Predict")
    parser.add_argument("--image",  required=True, help="Path to CT scan image")
    parser.add_argument("--model",  default=MODEL_PATH, help="Path to .pth weights")
    parser.add_argument("--tta",    action="store_true",
                        help="Use Test-Time Augmentation (more accurate, slower)")
    parser.add_argument("--json",   action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}"); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("       Train first: python model/train.py --demo"); sys.exit(1)

    device = get_device()
    model  = load_model(args.model, str(device)).to(device)
    result = predict(args.image, model, device, use_tta=args.tta)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    tta_tag = " [TTA]" if args.tta else ""
    print("\n" + "=" * 55)
    print(f"  KidneyScan AI — Prediction Result{tta_tag}")
    print("=" * 55)
    print(f"  Image     : {args.image}")
    print(f"  Predicted : {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Risk      : {result['severity']}")
    print(f"\n  {result['description']}")
    print("\n  Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<10} {bar:<30} {prob*100:5.1f}%")
    print("=" * 55)
    print("  Research only. Always consult a medical professional.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
