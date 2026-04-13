# models/model.py
# ResNet-18 backbone for Med-CPCL.
#
# THESIS REQUIREMENT (mandatory):
#   model.forward() MUST return (logits, latent_features)
#   latent_features = zi, stored in Score Memory (si, zi, ti) in plugin.py
#
# Architecture:
#   Input  : (B, 3, 28, 28)  -- BloodMNIST RGB
#   Backbone: ResNet-18 (modified for 28x28 input)
#   Latent : (B, 512)        -- penultimate layer, LATENT_DIM in config
#   Output : (B, num_classes) -- logits (NOT softmax)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, LATENT_DIM, DEVICE


class MedCPCLModel(nn.Module):
    """
    ResNet-18 adapted for 28x28 medical images.

    Key modification vs standard ResNet-18:
      - First conv: kernel 3x3, stride 1, no padding change
        (standard is 7x7 stride 2 -- too aggressive for 28x28)
      - MaxPool removed (would reduce 28x28 -> 6x6 before any learning)
      - This gives a 512-dim latent vector before the classifier head

    Returns:
        logits        : (B, num_classes)  -- raw scores for CE loss
        latent_features: (B, LATENT_DIM)  -- for Score Memory / drift comp
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()

        # Load ResNet-18 base
        backbone = models.resnet18(weights=None)  # no ImageNet weights for 28x28

        # ── Adapt for 28×28 input ─────────────────────────────────────────
        # Replace 7×7 conv (stride 2) with 3×3 conv (stride 1)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        # Remove maxpool (would shrink 28->13 before layer1)
        backbone.maxpool = nn.Identity()

        # ── Feature extractor (everything except final FC) ─────────────────
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,   # Identity — kept for structural clarity
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # (B, 512, 1, 1)
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Linear(LATENT_DIM, num_classes)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 28, 28)
        Returns:
            logits          : (B, num_classes)
            latent_features : (B, 512)   <-- zi in Score Memory
        """
        z = self.features(x)               # (B, 512, 1, 1)
        z = torch.flatten(z, 1)            # (B, 512)  -- latent_features
        logits = self.classifier(z)        # (B, num_classes)
        return logits, z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns only latent features (no grad by default)."""
        with torch.no_grad():
            _, z = self.forward(x)
        return z

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. Used in conformal scoring."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(num_classes: int = NUM_CLASSES,
                device: str = DEVICE) -> MedCPCLModel:
    """Factory function used by all training phases."""
    model = MedCPCLModel(num_classes=num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    import torch
    device = DEVICE

    model = build_model()
    model.eval()

    # ── Shape test ────────────────────────────────────────────────────────
    dummy = torch.randn(4, 3, 28, 28).to(device)
    logits, latent = model(dummy)

    print("=" * 50)
    print("  MedCPCLModel Architecture Verification")
    print("=" * 50)
    print(f"  Input  shape : {list(dummy.shape)}")
    print(f"  Logits shape : {list(logits.shape)}")
    print(f"  Latent shape : {list(latent.shape)}")
    print()

    # ── Parameter count ───────────────────────────────────────────────────
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print()

    # ── Sanity checks ─────────────────────────────────────────────────────
    assert logits.shape == (4, NUM_CLASSES),         f"Expected logits (4,{NUM_CLASSES}), got {logits.shape}"
    assert latent.shape == (4, LATENT_DIM),         f"Expected latent (4,{LATENT_DIM}), got {latent.shape}"

    proba = model.predict_proba(dummy)
    assert abs(proba.sum(dim=1).mean().item() - 1.0) < 1e-5,         "predict_proba does not sum to 1"

    z_only = model.get_latent(dummy)
    assert z_only.shape == (4, LATENT_DIM)

    print("  All assertions passed.")
    print(f"  Device: {next(model.parameters()).device}")
    print("=" * 50)
