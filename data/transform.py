"""Image preprocessing: Receipt image → processor pixel values tensor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image


def to_pixel_values(
    img_path: Path,
    processor: Callable[..., Any],
    size: tuple[int, int],
) -> Any:
    """Open image, resize to size, return processor pixel_values tensor."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size[0], size[1]))
    # processor is DonutProcessor or TrOCRProcessor; both accept PIL images
    result = processor(images=img, return_tensors="pt")
    return result.pixel_values
