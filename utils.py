"""
utils.py — Shared utilities for KidneyScan AI
==============================================
Functions used across training, evaluation and API modules.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, SEED


# ─── Reproducibility ──────────────────────────────────────────────────────────
def seed_everything(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Device ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_train_transform() -> transforms.Compose:
    """Data-augmentation transform for training."""
    return transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0] * 1.12), int(IMAGE_SIZE[1] * 1.12))),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation / inference."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transform(n_augments: int = 5):
    """
    Test-Time Augmentation — returns a list of transforms.
    Average predictions over multiple augmented views for improved accuracy.
    """
    return [get_train_transform() for _ in range(n_augments)]


# ─── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(image_source, use_tta: bool = False) -> torch.Tensor:
    """
    Load, resize and normalise an image.

    Args:
        image_source : path string OR PIL.Image OR bytes-like
        use_tta      : if True, return a batch with TTA augments
    Returns:
        Tensor of shape (1, 3, H, W) or (N, 3, H, W) for TTA
    """
    if isinstance(image_source, (str, os.PathLike)):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        import io
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image_source type: {type(image_source)}")

    tf = get_val_transform()
    tensor = tf(img).unsqueeze(0)   # (1, 3, H, W)

    if use_tta:
        tta = get_tta_transform()
        augments = torch.stack([t(img) for t in tta])  # (N, 3, H, W)
        tensor = torch.cat([tensor, augments], dim=0)

    return tensor


# ─── Model parameter counting ─────────────────────────────────────────────────
def count_parameters(model: torch.nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }


# ─── Plotting utilities ───────────────────────────────────────────────────────
_BG    = "#1a1a2e"
_PANEL = "#16213e"
_TEXT  = "white"
_COLORS = ["#e94560", "#53d8fb", "#f5a623", "#7ed321"]


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460")


def plot_training_history(history: dict, save_path: str) -> None:
    """Plot training & validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        _style_ax(ax)
        ax.grid(alpha=0.2)

    ep = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(ep, history["train_acc"], color="#e94560", lw=2, label="Train")
    axes[0].plot(ep, history["val_acc"],   color="#53d8fb", lw=2, ls="--", label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor=_BG, labelcolor=_TEXT)

    axes[1].plot(ep, history["train_loss"], color="#e94560", lw=2, label="Train")
    axes[1].plot(ep, history["val_loss"],   color="#53d8fb", lw=2, ls="--", label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor=_BG, labelcolor=_TEXT)

    plt.suptitle("KidneyScan AI — Training History", color=_TEXT, fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"[plot] Training history -> {save_path}")


def plot_confusion_matrix(cm, class_names: list, save_path: str) -> None:
    """Heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=_BG)
    _style_ax(ax, "Confusion Matrix")
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=30, ha="right", color=_TEXT)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, color=_TEXT)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", color=_TEXT)
    ax.set_xlabel("Predicted Label", color=_TEXT)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"[plot] Confusion matrix -> {save_path}")


def plot_roc_curves(y_true_onehot, y_pred_proba, class_names: list, save_path: str) -> None:
    """One-vs-Rest ROC curves."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=_BG)
    _style_ax(ax, "ROC-AUC Curves (One-vs-Rest)")
    for i, (cls, color) in enumerate(zip(class_names, _COLORS)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR", color=_TEXT); ax.set_ylabel("TPR", color=_TEXT)
    ax.legend(facecolor=_PANEL, labelcolor=_TEXT); ax.grid(alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"[plot] ROC curves -> {save_path}")


def plot_per_class_accuracy(cm, class_names: list, save_path: str) -> None:
    """Bar chart of per-class accuracy."""
    acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _style_ax(ax, "Per-Class Accuracy")
    bars = ax.bar(class_names, acc * 100, color=_COLORS, width=0.55, edgecolor="#0f3460", lw=1.5)
    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{a*100:.1f}%", ha="center", va="bottom", color=_TEXT, fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", color=_TEXT); ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"[plot] Per-class accuracy -> {save_path}")
