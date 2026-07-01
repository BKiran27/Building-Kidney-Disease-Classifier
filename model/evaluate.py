"""
Kidney Disease Detection — Model Evaluation (PyTorch)
======================================================
Usage:
  python evaluate.py --demo
  python evaluate.py --data ./data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from model import CLASS_NAMES, IMAGE_SIZE, load_model, get_device, IMAGENET_MEAN, IMAGENET_STD

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "kidney_model.pth")
BATCH_SIZE = 32

COLORS = ["#e94560", "#53d8fb", "#f5a623", "#7ed321"]
BG, PANEL, TEXT = "#1a1a2e", "#16213e", "white"


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _ax_style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460")


def plot_confusion_matrix(cm, save_dir):
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    _ax_style(ax, "Confusion Matrix")
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", color=TEXT)
    ax.set_yticks(tick_marks); ax.set_yticklabels(CLASS_NAMES, color=TEXT)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", color=TEXT)
    ax.set_xlabel("Predicted Label", color=TEXT)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"📊  Confusion matrix → {path}")


def plot_roc_curves(y_true_onehot, y_pred_proba, save_dir):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    _ax_style(ax, "ROC-AUC Curves (One-vs-Rest)")
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", color=TEXT); ax.set_ylabel("True Positive Rate", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=11)
    ax.grid(alpha=0.2, color="gray"); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"📊  ROC curves → {path}")


def plot_per_class_accuracy(cm, save_dir):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    _ax_style(ax, "Per-Class Accuracy")
    bars = ax.bar(CLASS_NAMES, per_class_acc * 100, color=COLORS, width=0.55, edgecolor="#0f3460", linewidth=1.5)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc*100:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", color=TEXT); ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.2, color="gray")
    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"📊  Per-class accuracy → {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Kidney Disease Detection — Evaluation")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--data", type=str)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Kidney Disease Detection — Evaluation (PyTorch)")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"❌  Model not found: {args.model}")
        print("    Train first: python train.py --demo")
        sys.exit(1)

    model = load_model(args.model, str(device)).to(device)
    print(f"✅  Model loaded")

    if args.demo:
        n    = 200
        X    = torch.rand(n, 3, *IMAGE_SIZE)
        y_true = torch.tensor([i % len(CLASS_NAMES) for i in range(n)])
        ds   = TensorDataset(X, y_true)
        loader = DataLoader(ds, batch_size=BATCH_SIZE)
    else:
        tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds     = datasets.ImageFolder(args.data, transform=tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=2)

    # ── Inference ───────────────────────────────────────────────────────────
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_pred_proba = np.concatenate(all_probs)
    y_true       = np.concatenate(all_labels)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    n_cls = len(CLASS_NAMES)
    y_true_onehot = np.eye(n_cls)[y_true]

    # ── Report ──────────────────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n─── Classification Report ───────────────────────────────────")
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    acc       = np.mean(y_true == y_pred)
    macro_auc = roc_auc_score(y_true_onehot, y_pred_proba, average="macro", multi_class="ovr")
    print(f"Overall Accuracy : {acc*100:.2f}%")
    print(f"Macro AUC        : {macro_auc:.4f}")

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_auc": float(macro_auc)}, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, OUTPUT_DIR)
    plot_roc_curves(y_true_onehot, y_pred_proba, OUTPUT_DIR)
    plot_per_class_accuracy(cm, OUTPUT_DIR)

    print(f"\n✅  All plots saved → {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
