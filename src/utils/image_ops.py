from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def load_image(path: Path) -> np.ndarray:
    return cv2.imread(str(path))


def resize_preserve(image: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale == 1.0:
        return image
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def image_statistics(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[1], image.shape[0]


def preprocess_image(image: np.ndarray, max_length: int = 1280) -> np.ndarray:
    """Apply contrast and noise reduction pipeline prior to OCR."""

    if image is None or image.size == 0:
        return image

    working = image.copy()

    # 1. Resize with longer edge capped to max_length using cubic interpolation.
    h, w = working.shape[:2]
    longest = max(h, w)
    if longest > max_length:
        scale = max_length / float(longest)
        new_size = (int(w * scale), int(h * scale))
        working = cv2.resize(working, new_size, interpolation=cv2.INTER_CUBIC)

    # 2. Normalize intensities per channel to stretch contrast.
    working = cv2.normalize(working, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 3. Denoise while preserving edges.
    working = cv2.fastNlMeansDenoisingColored(working, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    # 4. CLAHE in LAB color space
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. Deskew
    enhanced = deskew_image(enhanced)

    return enhanced


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskew the image using projection profile or contour angle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    
    if coords.size == 0:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Correct the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate using scipy for high quality interpolation
    if abs(angle) > 0.5: # Only rotate if skew is significant
        # order=3 is bicubic interpolation
        rotated = ndimage.rotate(image, angle, reshape=True, order=3, mode='constant', cval=255)
        return rotated
    return image


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Backward compatibility wrapper for legacy API."""
    return preprocess_image(image)


def to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def save_temp(image: np.ndarray, destination: Path) -> Path:
    Image.fromarray(image).save(destination)
    return destination


__all__ = [
    "load_image",
    "preprocess_image",
    "preprocess_for_ocr",
    "resize_preserve",
    "image_statistics",
    "to_pil",
    "save_temp",
    "deskew_image",
]
