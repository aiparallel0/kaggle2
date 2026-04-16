"""Image preprocessing: Receipt image → processor pixel values tensor."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def to_pixel_values(img_path: Path, processor: Callable[..., Any]) -> Any:
    """Open image; return processor pixel_values tensor.

    Args:
        img_path: Path to receipt image.
        processor: DonutProcessor or TrOCRProcessor that accepts PIL images.

    Returns:
        pixel_values tensor (1, C, H, W).
    """
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    result = processor(images=img, return_tensors="pt")
    return result.pixel_values
