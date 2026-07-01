"""
Kidney Disease Detection — Training Pipeline (PyTorch)
=======================================================
Supports two modes:
  --demo : Trains on synthetically generated data (no dataset needed)
  --data : Trains on the real CT-KIDNEY dataset folder

Usage:
  python train.py --demo
  python train.py --data ./data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    build_model, save_model, get_device,
    count_parameters, CLASS_NAMES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
MODEL_PATH   = os.path.join(BASE_DIR, "kidney_model.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")

# ─── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
VAL_SPLIT  = 0.2
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════════════
def get_train_transform():
    if not HAS_TORCHVISION:
        return None
    return transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0] * 1.1), int(IMAGE_SIZE[1] * 1.1))),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    if not HAS_TORCHVISION:
        return None
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Demo Dataset (synthetic)
# ══════════════════════════════════════════════════════════════════════════════
class SyntheticKidneyDataset(Dataset):
    """
    Synthetic dataset for demo purposes.
    Each class gets a slightly different mean pixel value so the model
    has something real to learn.
    """
    def __init__(self, n_per_class: int = 80):
        self.n_cls    = len(CLASS_NAMES)
        self.n_total  = n_per_class * self.n_cls
        self.images   = []
        self.labels   = []
        for cls_idx in range(self.n_cls):
            mean = 0.2 + cls_idx * 0.18
            for _ in range(n_per_class):
                img = np.clip(
                    np.random.normal(mean, 0.12, (3, *IMAGE_SIZE)), 0, 1
                ).astype(np.float32)
                self.images.append(torch.from_numpy(img))
                self.labels.append(cls_idx)

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ══════════════════════════════════════════════════════════════════════════════
def make_demo_loaders():
    print("⚙  DEMO MODE — using synthetic data (no real dataset required)")
    ds   = SyntheticKidneyDataset(n_per_class=100)
    n_val   = max(40, int(len(ds) * VAL_SPLIT))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train samples : {n_train}  |  Val samples : {n_val}\n")
    return train_loader, val_loader


def make_real_loaders(data_dir: str):
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision is required for real data loading.")
    full_ds = datasets.ImageFolder(data_dir, transform=get_train_transform())
    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    # Use validation transform for val split
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Classes found : {full_ds.classes}")
    print(f"  Train samples : {n_train}  |  Val samples : {n_val}\n")
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(history: dict, save_dir: str = OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f3460")

    epochs = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(epochs, history["train_acc"], color="#e94560", linewidth=2, label="Train Acc")
    axes[0].plot(epochs, history["val_acc"],   color="#53d8fb", linewidth=2, label="Val Acc", linestyle="--")
    axes[0].set_title("Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white"); axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, history["train_loss"], color="#e94560", linewidth=2, label="Train Loss")
    axes[1].plot(epochs, history["val_loss"],   color="#53d8fb", linewidth=2, label="Val Loss", linestyle="--")
    axes[1].set_title("Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white"); axes[1].grid(alpha=0.2)

    plt.suptitle("Kidney Disease Detection — Training History", color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊  Training plot saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Kidney Disease Detection — Training")
    mode   = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic demo data")
    mode.add_argument("--data", type=str, metavar="DIR", help="CT-KIDNEY dataset directory")
    parser.add_argument("--epochs", type=int,   default=EPOCHS,    help=f"Epochs (default {EPOCHS})")
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE, help=f"Batch size")
    parser.add_argument("--lr",     type=float, default=LR,         help="Learning rate")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Kidney Disease Detection — Training Pipeline (PyTorch)")
    print("=" * 60)
    print(f"  Device  : {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.demo:
        train_loader, val_loader = make_demo_loaders()
        max_epochs = min(args.epochs, 10)  # keep demo short
    else:
        if not os.path.isdir(args.data):
            print(f"❌  Directory not found: {args.data}"); sys.exit(1)
        train_loader, val_loader = make_real_loaders(args.data)
        max_epochs = args.epochs

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = build_model().to(device)
    params  = count_parameters(model)
    print(f"\n✅  Model built")
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}\n")

    # ── Optimizer & Loss ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, verbose=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    print("🚀  Starting training...\n")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader,   criterion, device)
        scheduler.step(va_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        flag = "🏆" if va_acc > best_val_acc else "  "
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_model(model, MODEL_PATH)

        print(f"  Epoch {epoch:02d}/{max_epochs}  "
              f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
              f"({elapsed:.1f}s)  {flag}")

    # ── Save history & plots ──────────────────────────────────────────────────
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📁  History saved → {HISTORY_PATH}")

    plot_history(history)

    print(f"\n🏆  Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("=" * 60)
    print("  Training complete! Model saved to:", MODEL_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
