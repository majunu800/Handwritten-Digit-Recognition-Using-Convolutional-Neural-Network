from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def load_image(image_input: ImageInput) -> Image.Image:
    """Load an image from path, uploaded file, NumPy array, or PIL object."""
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        return image.convert("RGB")

    if hasattr(image_input, "read"):
        image = Image.open(image_input)
        return image.convert("RGB")

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input.astype(np.uint8), mode="L")
        if image_input.ndim == 3:
            return Image.fromarray(image_input.astype(np.uint8))
        raise ValueError("Unsupported NumPy image shape. Expected 2D or 3D array.")

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    raise TypeError("Unsupported image input type.")


def prepare_mnist_image(image_input: ImageInput) -> np.ndarray:
    """Convert any image into a 28x28 grayscale array matching MNIST style."""
    pil_image = load_image(image_input)

    if pil_image.mode != "L":
        pil_image = ImageOps.grayscale(pil_image)

    image_array = np.array(pil_image)

    if image_array.size == 0:
        raise ValueError("The image is empty or could not be read.")

    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    # Remove very small empty regions or pure background areas.
    coords = np.column_stack(np.where(image_array > 10))
    if coords.size == 0:
        raise ValueError("No digit was detected in the image.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    padding = 8
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image_array.shape[0], y_max + padding + 1)
    x_max = min(image_array.shape[1], x_max + padding + 1)

    cropped = image_array[y_min:y_max, x_min:x_max]
    cropped_image = Image.fromarray(cropped.astype(np.uint8), mode="L")

    resized = cropped_image.resize((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    offset_x = (28 - 20) // 2
    offset_y = (28 - 20) // 2
    canvas.paste(resized, (offset_x, offset_y))

    normalized = np.array(canvas, dtype=np.float32) / 255.0
    normalized = normalized.reshape(28, 28, 1)
    return normalized


def preprocess_image_for_model(image_input: ImageInput) -> np.ndarray:
    """Return a batch-ready image with shape (1, 28, 28, 1) for the CNN."""
    img = prepare_mnist_image(image_input)
    return np.expand_dims(img, axis=0).astype(np.float32)
