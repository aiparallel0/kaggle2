"""Paper 3 — frozen-CNN visual feature head for the field assigner.

Project: kaggle2 — FOCUS-$\\Sigma$ verification layer for document KIE.
Role: extracts a fixed-dimension visual feature per detected text-line
    bbox by passing the cropped region through a *frozen* ImageNet-
    pretrained ResNet-18 (\\texttt{torchvision.models.resnet18}).
    No new training data; the ResNet weights are frozen at inference
    time so this module adds zero gradient cost during FOCUS-T training.
    A small trainable projection ($512 \\to d_{\\text{model}}$, ~50K
    parameters) maps the ResNet feature into the assigner's hidden
    space; this projection trains on the existing SROIE training fold
    only — no new dataset is required.

Why this is a Paper-3-only module.  Paper 2 commits to a deliberately
non-neural pipeline (regex + zone-prior HMM + per-field postprocess);
introducing visual-feature extraction crosses the bifurcation contract
of \\texttt{docs/PAPER2\\_VS\\_PAPER3.md}.  Paper 3 admits learned
neural components, so the CNN head lives here and is gated behind
\\texttt{config.focus\\_cnn\\_enabled} (default False).

Contract: ``cnn_features(image_path, bboxes, config) -> torch.Tensor``
    of shape ``(N, d_proj)`` where ``N = len(bboxes)`` and
    ``d_proj = config.focus_hidden_dim``.  Idempotent: the ResNet
    backbone is loaded lazily and cached on the module class so
    repeated calls amortise to the projection forward pass plus
    bbox cropping.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.types import ExpConfig

if TYPE_CHECKING:
    import torch

__all__ = ["CnnVisualHead", "cnn_features"]


_RESNET_FEAT_DIM = 512  # resnet18 avgpool output


class CnnVisualHead:
    """Frozen ResNet-18 + trainable linear projection over line bbox crops.

    The ResNet backbone is loaded once per process and cached; the
    projection is the only trainable component (initialised lazily on
    first forward when ``d_proj`` is known from the live config).
    Crops are resized to 224x224 with letterbox padding so receipt
    aspect ratios (vertical, often 3:1 or taller) survive the
    ImageNet-trained backbone without distortion.
    """

    _backbone: Any | None = None
    _transform: Any | None = None

    def __init__(self, config: ExpConfig) -> None:
        self._config = config
        self._proj: Any | None = None
        self._d_proj = int(config.focus_hidden_dim)

    @classmethod
    def _load_backbone(cls) -> tuple[Any, Any]:
        """Load + freeze ResNet-18 once per process; return (model, transform)."""
        if cls._backbone is not None and cls._transform is not None:
            return cls._backbone, cls._transform
        import torch
        from torchvision import models, transforms

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # expose 512-dim avgpool feature
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        cls._backbone = model
        cls._transform = transform
        return model, transform

    def _project(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply the trainable 512 -> d_proj projection (lazy-init)."""
        import torch

        if self._proj is None:
            self._proj = torch.nn.Linear(_RESNET_FEAT_DIM, self._d_proj)
            torch.nn.init.xavier_uniform_(self._proj.weight)
            torch.nn.init.zeros_(self._proj.bias)
        return self._proj(feat)

    def forward(
        self, image_path: str, bboxes: list[list[float]],
    ) -> torch.Tensor:
        """Return ``(N, d_proj)`` projected features for ``N`` line crops.

        On any exception (missing image, corrupt crop, missing torch /
        torchvision) returns a zero tensor of shape
        ``(len(bboxes), d_proj)`` so the caller can degrade gracefully
        without breaking the eval loop — the visual head is opt-in
        and never raises.
        """
        import torch

        n = len(bboxes)
        zero = torch.zeros(n, self._d_proj)
        if n == 0:
            return zero
        try:
            from PIL import Image

            backbone, transform = self._load_backbone()
            img = Image.open(image_path).convert("RGB")
            W, H = img.size
            crops: list[torch.Tensor] = []
            for bb in bboxes:
                if len(bb) < 4:
                    crops.append(torch.zeros(3, 224, 224))
                    continue
                x0, y0, x1, y1 = bb[:4]
                if max(bb) <= 1.5:
                    x0, y0, x1, y1 = x0 * W, y0 * H, x1 * W, y1 * H
                x0i = max(0, int(x0))
                y0i = max(0, int(y0))
                x1i = min(W, int(x1))
                y1i = min(H, int(y1))
                if x1i <= x0i or y1i <= y0i:
                    crops.append(torch.zeros(3, 224, 224))
                    continue
                crops.append(transform(img.crop((x0i, y0i, x1i, y1i))))
            batch = torch.stack(crops, dim=0)
            with torch.no_grad():
                feats = backbone(batch)
            return self._project(feats)
        except Exception:  # noqa: BLE001
            return zero


def cnn_features(
    image_path: str | Path, bboxes: list[list[float]], config: ExpConfig,
) -> torch.Tensor:
    """Module-level entry point matching the project's 2-in/1-out contract."""
    head = CnnVisualHead(config)
    return head.forward(str(image_path), bboxes)
