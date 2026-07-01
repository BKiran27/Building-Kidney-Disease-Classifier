"""
model.py — KidneyScan AI  (v2 — High-Accuracy Architecture)
==============================================================
Upgrades over v1:
  • CBAM (Channel + Spatial Attention) after backbone features
  • GeM (Generalised Mean) Pooling — more powerful than AvgPool
  • Deeper, regularised classification head with Residual skip
  • Drop-path / stochastic depth support
  • `predict_proba` supports Test-Time Augmentation (TTA)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Make project root importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class GeM(nn.Module):
    """
    Generalised Mean Pooling.
    Learns the optimal pooling exponent `p` during training.
    Outperforms AvgPool for image classification & retrieval.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.data.item():.4f})"


class ChannelAttention(nn.Module):
    """CBAM channel attention — squeeze-and-excitation style."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    """CBAM spatial attention — where to look."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        cat     = torch.cat([avg_out, max_out], dim=1)
        scale   = self.sigmoid(self.conv(cat))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention → spatial attention in sequence.
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel  = ChannelAttention(in_channels, reduction)
        self.spatial  = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResidualHead(nn.Module):
    """
    Classification head with a residual (skip) connection.
    Structure:
        FC(2048 → 1024) + BN + GELU + Drop(0.5)  → FC(1024 → 512) + BN + GELU + Drop(0.4)
        + Residual shortcut: FC(2048 → 512)        → FC(512 → num_classes)
    """
    def __init__(self, in_features: int = 2048, num_classes: int = NUM_CLASSES,
                 drop1: float = 0.5, drop2: float = 0.4):
        super().__init__()
        mid = 1024
        out = 512

        # Main path
        self.main = nn.Sequential(
            nn.Linear(in_features, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(drop1),
            nn.Linear(mid, out),
            nn.BatchNorm1d(out),
            nn.GELU(),
            nn.Dropout(drop2),
        )
        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Linear(in_features, out, bias=False),
            nn.BatchNorm1d(out),
        )
        # Final projection
        self.classifier = nn.Linear(out, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.main(x) + self.shortcut(x))


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════

class KidneyDiseaseNet(nn.Module):
    """
    High-accuracy kidney disease classifier.

    Architecture:
        ResNet50 backbone (3-phase progressive fine-tuning)
        → CBAM attention (channel + spatial)
        → GeM Pooling
        → ResidualHead (1024 → 512 + shortcut, then → 4 classes)

    Freezing strategy (handled by train.py):
        Phase 1: all backbone frozen,  head trainable
        Phase 2: last 2 ResNet blocks + head trainable
        Phase 3: entire network trainable (very small LR)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove final FC + AvgPool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4, (B, 2048, H, W)

        # Freeze everything initially (caller unfreezes progressively)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── Attention ─────────────────────────────────────────────────────
        self.cbam = CBAM(in_channels=2048, reduction=16, kernel_size=7)

        # ── Pooling ───────────────────────────────────────────────────────
        self.pool = GeM(p=3.0)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = ResidualHead(in_features=2048, num_classes=num_classes)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.backbone(x)          # (B, 2048, 7, 7)
        feat   = self.cbam(feat)           # attention-weighted features
        pooled = self.pool(feat).flatten(1) # (B, 2048)
        return self.head(pooled)           # (B, num_classes)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
        """
        Return class probabilities.
        If tta=True, x is assumed to be a batch of augmented views;
        probabilities are averaged across all views.
        """
        self.eval()
        if tta and x.dim() == 4 and x.shape[0] > 1:
            probs = torch.softmax(self(x), dim=1)   # (N_aug, C)
            return probs.mean(dim=0, keepdim=True)  # (1, C)
        return torch.softmax(self(x), dim=1)

    # ── Phased unfreezing ─────────────────────────────────────────────────
    def freeze_backbone(self):
        """Phase 1: freeze entire backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n_blocks: int = 2):
        """
        Phase 2: unfreeze the last `n_blocks` residual layers.
        ResNet50 blocks: layer1, layer2, layer3, layer4  (indices 4-7 in children)
        """
        children = list(self.backbone.children())
        for child in children[-n_blocks:]:
            for p in child.parameters():
                if not isinstance(child, nn.BatchNorm2d):
                    p.requires_grad = True

    def unfreeze_all(self):
        """Phase 3: unfreeze everything."""
        for p in self.parameters():
            p.requires_grad = True
        # Keep BN layers frozen (they carry ImageNet statistics)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES) -> KidneyDiseaseNet:
    return KidneyDiseaseNet(num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def save_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[save] Model -> {path}")


def load_model(path: str, device: str = "cpu") -> KidneyDiseaseNet:
    model = KidneyDiseaseNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = get_device()
    m = build_model().to(device)
    p = count_parameters(m)
    print(f"Device              : {device}")
    print(f"Total parameters    : {p['total']:,}")
    print(f"Trainable (phase 1) : {p['trainable']:,}")

    # Phase 2 test
    m.unfreeze_top_blocks(2)
    p2 = count_parameters(m)
    print(f"Trainable (phase 2) : {p2['trainable']:,}")

    # Phase 3 test
    m.unfreeze_all()
    p3 = count_parameters(m)
    print(f"Trainable (phase 3) : {p3['trainable']:,}")

    dummy = torch.randn(2, 3, *IMAGE_SIZE).to(device)
    out   = m(dummy)
    print(f"Output shape        : {out.shape}")
    probs = torch.softmax(out, dim=1)
    print(f"Softmax sums        : {probs.sum(dim=1).tolist()}")
    print("Architecture OK!")
