from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np
import cv2

from src.io_utils import list_images, load_image

def _ensure_32(img: np.ndarray, size: int = 32) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def build_tile_index(
    tiles_dir: Union[str, Path],
    size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Load all images under tiles_dir and build a tile DB with:
      - images: (N, H, W, 3) uint8
      - mean_rgb: (N, 3) float32
      - flat_f32: (N, H*W*3) float32 in [0,1]  (for MSE matching)
      - paths: list[str]
    """
    tiles_dir = Path(tiles_dir)
    paths = list_images(tiles_dir)
    imgs: List[np.ndarray] = []
    keep: List[str] = []
    for p in paths:
        try:
            img = load_image(p)
            img = _ensure_32(img, size=size)
            imgs.append(img)
            keep.append(str(p))
        except Exception:
            continue

    if not imgs:
        raise RuntimeError(f"No tiles found in {tiles_dir}")

    images = np.stack(imgs, axis=0)  # (N, 32, 32, 3)
    mean_rgb = images.astype(np.float32).mean(axis=(1,2))  # (N,3)
    flat_f32 = (images.astype(np.float32) / 255.0).reshape(images.shape[0], -1)

    return {
        "images": images,          # uint8
        "mean_rgb": mean_rgb,      # float32
        "flat_f32": flat_f32,      # float32 [0,1]
        "paths": np.array(keep),   # for debugging
        "tile_size": np.array([size], dtype=np.int32),
    }
