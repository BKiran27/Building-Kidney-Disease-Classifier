"""
train.py — KidneyScan AI  (v2 — High-Accuracy Training Pipeline)
==================================================================
Accuracy-boosting upgrades over v1:
  1. 3-Phase Progressive Unfreezing  — head only → top-2 blocks → full network
  2. Cosine Annealing LR with Warmup — smooth, optimal LR schedule
  3. Label Smoothing Loss            — prevents overconfidence, improves calibration
  4. MixUp Augmentation             — powerful regulariser (+3-5% accuracy)
  5. Focal Loss option              — handles class imbalance
  6. EMA (Exponential Moving Average)— stable, better-generalising weights
  7. Gradient Clipping              — prevents exploding gradients
  8. Advanced data augmentation     — stronger transforms for medical images
  9. Best checkpoint saving         — saves both best val_acc AND best val_loss

Usage:
  python model/train.py --demo                      # synthetic data, fast test
  python model/train.py --data ./data/CT-KIDNEY-... # real Kaggle dataset
"""

import os
import sys
import math
import json
import time
import argparse
import copy
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchvision import datasets, transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, SEED,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    VAL_SPLIT, LR_FACTOR, LR_PATIENCE, LR_MIN,
    OUTPUT_DIR, MODEL_PATH,
)
from model.model import build_model, save_model, get_device, count_parameters

# ─── Seed ─────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Paths ─────────────────────────────────────────────────────────────────────
HISTORY_PATH    = os.path.join(OUTPUT_DIR, "history.json")
BEST_ACC_PATH   = os.path.join(os.path.dirname(MODEL_PATH), "kidney_model_best_acc.pth")
BEST_LOSS_PATH  = os.path.join(os.path.dirname(MODEL_PATH), "kidney_model_best_loss.pth")


# ══════════════════════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Smoothing=0.1 means 10% of the probability mass is spread uniformly.
    Prevents overconfidence and typically yields +0.5-1.5% accuracy.
    """
    def __init__(self, smoothing: float = 0.1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        # Hard label part
        nll_loss  = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        # Smooth label part
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard ones.
    Great for imbalanced datasets.
    γ=2 is the standard choice.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ══════════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average of weights)
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Maintains an exponential moving average of model weights.
    EMA weights are more stable and typically improve validation accuracy.
    decay=0.9999 is standard for large datasets; 0.999 for small.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay     = decay
        self.shadow    = {}
        self.original  = {}
        # Initialise shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] = (
                        self.decay * self.shadow[name]
                        + (1.0 - self.decay) * param.data
                    )

    def apply_shadow(self, model: nn.Module):
        """Swap in EMA weights (call before evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original:
                param.data.copy_(self.original[name])
        self.original.clear()


# ══════════════════════════════════════════════════════════════════════════════
# MixUp Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, device=None):
    """
    Returns mixed inputs, pairs of targets, and mixing coefficient λ.
    alpha=0.4 is a good default for medical imaging.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════════════════════
# LR Scheduler — Cosine Annealing with Linear Warmup
# ══════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                                     min_lr_ratio: float = 0.01):
    """
    Linear warmup → cosine decay.
    Better than ReduceLROnPlateau for training from scratch / fine-tuning.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def get_strong_train_transform():
    """Medical-imaging-grade augmentation pipeline."""
    if not HAS_TV:
        return None
    return transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0] * 1.15), int(IMAGE_SIZE[1] * 1.15))),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_val_transform():
    if not HAS_TV:
        return None
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SyntheticKidneyDataset(Dataset):
    """Synthetic dataset for demo/CI — no real data needed."""
    def __init__(self, n_per_class: int = 120):
        self.items = []
        for cls_idx in range(NUM_CLASSES):
            mean = 0.2 + cls_idx * 0.18
            for _ in range(n_per_class):
                img = np.clip(
                    np.random.normal(mean, 0.12, (3, *IMAGE_SIZE)), 0, 1
                ).astype(np.float32)
                self.items.append((torch.from_numpy(img), cls_idx))

    def __len__(self):  return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_demo_loaders(batch_size: int):
    ds    = SyntheticKidneyDataset(n_per_class=120)
    n_val = max(40, int(len(ds) * VAL_SPLIT))
    n_tr  = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[demo] Synthetic  train={n_tr}  val={n_val}")
    return train_loader, val_loader


def make_real_loaders(data_dir: str, batch_size: int):
    full = datasets.ImageFolder(data_dir, transform=get_strong_train_transform())
    n_val = int(len(full) * VAL_SPLIT)
    n_tr  = len(full) - n_val
    tr, va = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))

    # Val subset gets deterministic transform
    val_ds = datasets.ImageFolder(data_dir, transform=get_val_transform())
    # Replace underlying dataset for val split
    va.dataset = val_ds

    workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    print(f"[real] Classes: {full.classes}")
    print(f"[real] train={n_tr}  val={n_val}")
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup: bool = True, mixup_alpha: float = 0.4,
                    max_grad_norm: float = 5.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            logits = model(images)
            loss   = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            # Approximate accuracy for mixed labels
            preds   = logits.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════════════
# 3-Phase Training
# ══════════════════════════════════════════════════════════════════════════════

def run_phase(model, train_loader, val_loader, criterion, device,
              phase_name: str, epochs: int, lr: float, weight_decay: float,
              use_mixup: bool, use_ema: bool, warmup_epochs: int,
              best_acc: float, best_loss: float,
              history: dict, ema: "EMA | None") -> tuple:
    """
    Run one training phase. Returns (best_acc, best_loss, updated_ema).
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)

    if use_ema and ema is None:
        ema = EMA(model, decay=0.9995)

    print(f"\n{'─'*60}")
    print(f"  Phase: {phase_name}  |  epochs={epochs}  lr={lr:.2e}  mixup={use_mixup}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup,
        )

        # Evaluate with EMA weights if available
        if use_ema and ema is not None:
            ema.apply_shadow(model)

        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        if use_ema and ema is not None:
            ema.restore(model)
            ema.update(model)

        scheduler.step()
        elapsed = time.time() - t0

        # Track best checkpoints
        acc_flag  = " 🏆" if va_acc  > best_acc  else ""
        loss_flag = " 📉" if va_loss < best_loss else ""
        if va_acc > best_acc:
            best_acc = va_acc
            save_model(model, BEST_ACC_PATH)
        if va_loss < best_loss:
            best_loss = va_loss
            save_model(model, BEST_LOSS_PATH)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"  [{phase_name}] ep {epoch:02d}/{epochs}  "
              f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
              f"({elapsed:.1f}s){acc_flag}{loss_flag}")

    return best_acc, best_loss, ema


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BG, PANEL = "#1a1a2e", "#16213e"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white"); ax.grid(alpha=0.2)
        for sp in ax.spines.values(): sp.set_edgecolor("#0f3460")

    ep = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(ep, history["train_acc"], color="#e94560", lw=2, label="Train")
    axes[0].plot(ep, history["val_acc"],   color="#53d8fb", lw=2, ls="--", label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(facecolor=BG, labelcolor="white")

    axes[1].plot(ep, history["train_loss"], color="#e94560", lw=2, label="Train")
    axes[1].plot(ep, history["val_loss"],   color="#53d8fb", lw=2, ls="--", label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(facecolor=BG, labelcolor="white")

    if history.get("lr"):
        axes[2].plot(ep, history["lr"], color="#f5a623", lw=2)
        axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")

    plt.suptitle("KidneyScan AI v2 — Training History", color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[plot] Training history -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="KidneyScan AI v2 — Training")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Synthetic demo data (fast)")
    mode.add_argument("--data", type=str, metavar="DIR",
                      help="Path to CT-KIDNEY dataset directory")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help=f"Total training epochs (default {EPOCHS})")
    p.add_argument("--batch",  type=int, default=BATCH_SIZE)
    p.add_argument("--lr",     type=float, default=LEARNING_RATE)
    p.add_argument("--no-mixup",  action="store_true", help="Disable MixUp augmentation")
    p.add_argument("--no-ema",    action="store_true", help="Disable EMA weights")
    p.add_argument("--focal",     action="store_true", help="Use Focal Loss instead of Label Smoothing CE")
    p.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing factor (default 0.1)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  KidneyScan AI v2 — High-Accuracy Training Pipeline")
    print("=" * 60)
    print(f"  Device  : {device}")
    print(f"  MixUp   : {not args.no_mixup}")
    print(f"  EMA     : {not args.no_ema}")
    print(f"  Loss    : {'Focal' if args.focal else f'LabelSmoothing(={args.smoothing})'}")

    # ── Data ────────────────────────────────────────────────────────────────
    demo_mode = args.demo
    if demo_mode:
        train_loader, val_loader = make_demo_loaders(args.batch)
        total_epochs = min(args.epochs, 12)   # keep demo fast
        # Phases: (name, epochs, lr_multiplier, use_mixup)
        phases = [
            ("Head-only",    3,  1.0,    False),
            ("Top-2-blocks", 5,  0.1,    True),
            ("Full-net",     4,  0.01,   True),
        ]
    else:
        if not os.path.isdir(args.data):
            print(f"ERROR: Directory not found: {args.data}"); sys.exit(1)
        train_loader, val_loader = make_real_loaders(args.data, args.batch)
        total_epochs = args.epochs
        ep1 = max(3, total_epochs // 6)
        ep2 = max(5, total_epochs // 3)
        ep3 = total_epochs - ep1 - ep2
        phases = [
            ("Head-only",    ep1, 1.0,  False),
            ("Top-2-blocks", ep2, 0.1,  not args.no_mixup),
            ("Full-net",     ep3, 0.02, not args.no_mixup),
        ]

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    model.freeze_backbone()
    p = count_parameters(model)
    print(f"\n  Total params    : {p['total']:,}")
    print(f"  Trainable(ph.1) : {p['trainable']:,}\n")

    # ── Loss ────────────────────────────────────────────────────────────────
    if args.focal:
        criterion = FocalLoss(gamma=2.0).to(device)
    else:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=args.smoothing, num_classes=NUM_CLASSES
        ).to(device)

    # ── State ────────────────────────────────────────────────────────────────
    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc   = 0.0
    best_loss  = float("inf")
    ema        = None

    # ══════════════════════════════════════════════════════════════════════
    # 3-Phase Training
    # ══════════════════════════════════════════════════════════════════════
    for ph_name, ph_epochs, lr_mul, ph_mixup in phases:
        # Unfreeze appropriate layers
        if ph_name == "Top-2-blocks":
            model.unfreeze_top_blocks(2)
        elif ph_name == "Full-net":
            model.unfreeze_all()

        phase_lr = args.lr * lr_mul
        warmup   = max(1, ph_epochs // 5)

        best_acc, best_loss, ema = run_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            phase_name=ph_name,
            epochs=ph_epochs,
            lr=phase_lr,
            weight_decay=WEIGHT_DECAY,
            use_mixup=ph_mixup and not args.no_mixup,
            use_ema=not args.no_ema,
            warmup_epochs=warmup,
            best_acc=best_acc,
            best_loss=best_loss,
            history=history,
            ema=ema,
        )

    # ── Final save (last weights) ─────────────────────────────────────────
    save_model(model, MODEL_PATH)

    # ── Save history ─────────────────────────────────────────────────────
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[save] History -> {HISTORY_PATH}")

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_history(history)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy : {best_acc*100:.2f}%")
    print(f"  Best Val Loss     : {best_loss:.4f}")
    print(f"  Best-acc model    : {BEST_ACC_PATH}")
    print(f"  Best-loss model   : {BEST_LOSS_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
