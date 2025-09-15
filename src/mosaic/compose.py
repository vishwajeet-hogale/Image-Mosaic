# src/mosaic/compose.py
from __future__ import annotations
from typing import Literal, Tuple, Dict
import numpy as np
import cv2

from ..tiling.grid import infer_grid, block_view
from ..tiling.matchers import match_by_mean_rgb, match_by_mse

Metric = Literal["mean_rgb", "mse"]
Classifier = Literal["none", "luma_bins", "rgb_bins"]

def _ensure_grid_compatible(img_rgb: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    new_h = (h // tile_size) * tile_size
    new_w = (w // tile_size) * tile_size
    if new_h == h and new_w == w:
        return img_rgb
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return img_rgb[y0:y0+new_h, x0:x0+new_w]

def _flatten_blocks(blocks: np.ndarray) -> np.ndarray:
    R, C, ts, _, _ = blocks.shape
    return (blocks.reshape(R * C, ts * ts * 3).astype(np.float32) / 255.0)

def _means_from_blocks(blocks: np.ndarray) -> np.ndarray:
    mean_rgb = blocks.mean(axis=(2, 3), dtype=np.float32)  # (R,C,3)
    R, C, _ = mean_rgb.shape
    return mean_rgb.reshape(R * C, 3)  # (K,3)

def _luma_from_rgb(rgb: np.ndarray) -> np.ndarray:
    # rgb can be (...,3). expects 0..255 scale floats
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

def _digitize(values: np.ndarray, n_bins: int) -> np.ndarray:
    # uniform bins in [0,255]
    edges = np.linspace(0.0, 255.0, num=n_bins + 1, dtype=np.float32)
    # np.digitize returns 1..n_bins; shift to 0..n_bins-1
    return np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1).astype(np.int32)

def _rgb_bins_indices(mean_rgb: np.ndarray, base: int) -> np.ndarray:
    edges = np.linspace(0.0, 255.0, num=base + 1, dtype=np.float32)
    r = np.clip(np.digitize(mean_rgb[..., 0], edges[1:-1]), 0, base - 1)
    g = np.clip(np.digitize(mean_rgb[..., 1], edges[1:-1]), 0, base - 1)
    b = np.clip(np.digitize(mean_rgb[..., 2], edges[1:-1]), 0, base - 1)
    return (r * (base * base) + g * base + b).astype(np.int32)

def _classify_patches(mean_rgb_patches: np.ndarray, classifier: Classifier, luma_bins: int, rgb_bins: int) -> np.ndarray:
    if classifier == "none":
        return None
    if classifier == "luma_bins":
        luma = _luma_from_rgb(mean_rgb_patches)
        return _digitize(luma, n_bins=luma_bins)  # (K,)
    if classifier == "rgb_bins":
        return _rgb_bins_indices(mean_rgb_patches, base=rgb_bins)  # (K,)
    raise ValueError(f"Unknown classifier: {classifier}")

def _classify_tiles(tile_db: Dict[str, np.ndarray], classifier: Classifier, luma_bins: int, rgb_bins: int) -> np.ndarray:
    if classifier == "none":
        return None
    mean_rgb_tiles = tile_db["mean_rgb"]  # (N,3) float32 in 0..255
    if classifier == "luma_bins":
        luma = _luma_from_rgb(mean_rgb_tiles)
        return _digitize(luma, n_bins=luma_bins)  # (N,)
    if classifier == "rgb_bins":
        return _rgb_bins_indices(mean_rgb_tiles, base=rgb_bins)  # (N,)
    raise ValueError(f"Unknown classifier: {classifier}")

def mosaic_from_uploaded(
    img_rgb: np.ndarray,
    tile_db: Dict[str, np.ndarray],
    tile_size: int = 32,
    metric: Metric = "mean_rgb",
    classifier: Classifier = "none",
    luma_bins: int = 8,
    rgb_bins: int = 4,
    blend: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a mosaic for the uploaded image.
    - classifier == "none": global matching (fast)
    - classifier == "luma_bins" or "rgb_bins": match within each class
    Returns (mosaic_rgb, index_map[R,C]).
    """
    img_rgb = _ensure_grid_compatible(img_rgb, tile_size)
    gs = infer_grid(img_rgb, tile_size)                 # rows, cols
    blocks = block_view(img_rgb, tile_size)             # (R,C,ts,ts,3)
    R, C = gs.rows, gs.cols
    K = R * C

    # Prepare patch features
    patch_means = _means_from_blocks(blocks)            # (K,3) float32 0..255
    if metric == "mse":
        patch_flat = _flatten_blocks(blocks)            # (K,D) float32 0..1

    # Optional classification
    patch_cat = _classify_patches(patch_means, classifier, luma_bins, rgb_bins)  # (K,) or None
    tile_cat  = _classify_tiles(tile_db, classifier, luma_bins, rgb_bins)        # (N,) or None

    idx_flat = np.empty((K,), dtype=np.int32)

    if classifier == "none":
        if metric == "mean_rgb":
            idx_flat[:] = match_by_mean_rgb(patch_means, tile_db["mean_rgb"])
        else:
            idx_flat[:] = match_by_mse(patch_flat, tile_db["flat_f32"])
    else:
        # group patches by category and match only to tiles in the same category
        unique_cats = np.unique(patch_cat)
        for cat in unique_cats:
            p_idx = np.nonzero(patch_cat == cat)[0]
            t_idx = np.nonzero(tile_cat == cat)[0]
            if t_idx.size == 0:
                # Fallback to global pool if this category has no tiles
                if metric == "mean_rgb":
                    idx_flat[p_idx] = match_by_mean_rgb(patch_means[p_idx], tile_db["mean_rgb"])
                else:
                    idx_flat[p_idx] = match_by_mse(patch_flat[p_idx], tile_db["flat_f32"])
            else:
                if metric == "mean_rgb":
                    best_local = match_by_mean_rgb(patch_means[p_idx], tile_db["mean_rgb"][t_idx])
                else:
                    best_local = match_by_mse(patch_flat[p_idx], tile_db["flat_f32"][t_idx])
                idx_flat[p_idx] = t_idx[best_local]

    # Compose output image from chosen tiles
    tiles = tile_db["images"]                           # (N, ts, ts, 3) uint8
    idx_map = idx_flat.reshape(R, C)                    # (R,C)
    tile_block = tiles[idx_map]                         # (R,C,ts,ts,3)
    mosaic = tile_block.transpose(0,2,1,3,4).reshape(R*tile_size, C*tile_size, 3)

    if blend > 0.0:
        mosaic = cv2.addWeighted(mosaic.astype(np.float32), 1.0 - blend,
                                 img_rgb.astype(np.float32), blend, 0.0).astype(np.uint8)

    return mosaic, idx_map
