"""
Kidney Disease Detection Model (PyTorch)
=========================================
Uses ResNet50 transfer learning to classify kidney CT scans into:
  - Cyst
  - Normal
  - Stone
  - Tumor
"""

import torch
import torch.nn as nn
from torchvision import models


# ─── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Model Definition ──────────────────────────────────────────────────────────
class KidneyDiseaseNet(nn.Module):
    """
    ResNet50-based kidney disease classifier.

    Architecture:
        ResNet50 backbone (fine-tune last 2 blocks)
        → Adaptive Avg Pool
        → FC(2048 → 512) + BN + ReLU + Dropout(0.4)
        → FC(512 → 256)  + BN + ReLU + Dropout(0.3)
        → FC(256 → 4, Softmax)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, unfreeze_last_n: int = 2):
        super().__init__()

        # ── Load pretrained ResNet50 ────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze all layers first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last `unfreeze_last_n` residual layers (layer3, layer4 …)
        layers_to_unfreeze = list(backbone.children())[-unfreeze_last_n - 1: -1]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Remove the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to AvgPool

        # ── Custom classification head ─────────────────────────────────────
        in_features = 2048  # ResNet50 output size

        self.head = nn.Sequential(
            nn.Flatten(),
            # Block 1
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            # Output
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, 2048, 1, 1)
        logits   = self.head(features) # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        with torch.no_grad():
            return torch.softmax(self(x), dim=1)


def build_model(num_classes: int = NUM_CLASSES) -> KidneyDiseaseNet:
    """Build and return the KidneyDiseaseNet model."""
    return KidneyDiseaseNet(num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def save_model(model: nn.Module, path: str) -> None:
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"💾  Model saved → {path}")


def load_model(path: str, device: str = "cpu") -> KidneyDiseaseNet:
    """Load a saved model from disk."""
    model = KidneyDiseaseNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    model  = build_model().to(device)
    params = count_parameters(model)
    print(f"Total parameters    : {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen  parameters  : {params['frozen']:,}")

    # Test forward pass
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"Output shape        : {out.shape}")  # (2, 4)
    probs = torch.softmax(out, dim=1)
    print(f"Softmax sum         : {probs.sum(dim=1)}")  # should be ~[1., 1.]
