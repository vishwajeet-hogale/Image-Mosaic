# src/tiling/grid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import cv2

@dataclass(frozen=True)
class GridSpec:
    tile_size: int
    rows: int
    cols: int
    height: int
    width: int

def infer_grid(img_rgb: np.ndarray, tile_size: int) -> GridSpec:
    """
    Ensure the image height/width are multiples of tile_size and return grid spec.
    Assumes you've already resized/cropped appropriately; asserts to catch mistakes.
    """
    h, w = img_rgb.shape[:2]
    assert h % tile_size == 0 and w % tile_size == 0, (
        f"Image dims ({h}x{w}) must be multiples of tile_size={tile_size}. "
        "Use your Step 1 resize/crop or the _ensure_grid_compatible() in compose.py."
    )
    rows = h // tile_size
    cols = w // tile_size
    return GridSpec(tile_size=tile_size, rows=rows, cols=cols, height=h, width=w)

def block_view(img: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Vectorized grid view (no Python loops).
    Returns shape (rows, cols, tile_size, tile_size, C) for 3D images
    or (rows, cols, tile_size, tile_size) for 2D.
    """
    h, w = img.shape[:2]
    rows = h // tile_size
    cols = w // tile_size
    if img.ndim == 3:
        C = img.shape[2]
        v = img.reshape(rows, tile_size, cols, tile_size, C)
        v = np.swapaxes(v, 1, 2)  # -> (rows, cols, tile, tile, C)
        return v
    else:
        v = img.reshape(rows, tile_size, cols, tile_size)
        v = np.swapaxes(v, 1, 2)
        return v

def per_cell_stats(img_rgb: np.ndarray, tile_size: int) -> Dict[str, np.ndarray]:
    """
    Compute per-cell mean/std in RGB and mean luma (BT.601) using a vectorized view.
    """
    gs = infer_grid(img_rgb, tile_size)
    tiles = block_view(img_rgb, tile_size)  # (R,C,ts,ts,3)
    mean_rgb = tiles.mean(axis=(2, 3), dtype=np.float32)  # (R,C,3)
    std_rgb  = tiles.std(axis=(2, 3), dtype=np.float32)   # (R,C,3)
    mean_luma = (0.299 * mean_rgb[..., 0] +
                 0.587 * mean_rgb[..., 1] +
                 0.114 * mean_rgb[..., 2]).astype(np.float32)  # (R,C)
    return {
        "grid_spec": np.array([gs.rows, gs.cols, gs.tile_size], dtype=np.int32),
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb,
        "mean_luma": mean_luma,
    }

def draw_grid_overlay(img_rgb: np.ndarray, tile_size: int, color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    """
    Draw thin grid lines to visualize the tiling (for UI/report).
    """
    h, w = img_rgb.shape[:2]
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    for y in range(tile_size, h, tile_size):
        cv2.line(out_bgr, (0, y), (w, y), bgr, thickness)
    for x in range(tile_size, w, tile_size):
        cv2.line(out_bgr, (x, 0), (x, h), bgr, thickness)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
