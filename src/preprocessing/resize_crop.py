from typing import Optional, Tuple
import numpy as np
import cv2

def _enforce_multiple(x: int, m: int) -> int:
    """Round down x to nearest multiple of m."""
    return (x // m) * m

def compute_target_size(
    h: int,
    w: int,
    tile_size: int,
    grid_cols: Optional[int] = None,
    grid_rows: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Decide a target (H, W) so that both are multiples of tile_size.
    If grid_cols is provided, width becomes grid_cols * tile_size, height is scaled to preserve aspect.
    If grid_rows is provided, height becomes grid_rows * tile_size, width is scaled to preserve aspect.
    If neither provided, rounds original dims down to nearest multiple of tile_size.
    """
    aspect = w / max(h, 1)
    if grid_cols is not None:
        target_w = grid_cols * tile_size
        target_h = int(round(target_w / max(aspect, 1e-6)))
    elif grid_rows is not None:
        target_h = grid_rows * tile_size
        target_w = int(round(target_h * aspect))
    else:
        target_h, target_w = h, w

    # ensure both are multiples of tile_size
    target_h = max(tile_size, _enforce_multiple(target_h, tile_size))
    target_w = max(tile_size, _enforce_multiple(target_w, tile_size))
    return target_h, target_w

def resize_and_center_crop_to_grid(
    img_rgb: np.ndarray,
    tile_size: int,
    grid_cols: Optional[int] = None,
    grid_rows: Optional[int] = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize with aspect preserved, then center-crop so final dims are multiples
    of tile_size AND match requested grid if provided.
    """
    h, w = img_rgb.shape[:2]
    target_h, target_w = compute_target_size(h, w, tile_size, grid_cols, grid_rows)

    # First scale so that both dims are >= target dims (so we can crop to target)
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interpolation)

    # Center-crop to exact target dims
    y0 = max((new_h - target_h) // 2, 0)
    x0 = max((new_w - target_w) // 2, 0)
    cropped = resized[y0 : y0 + target_h, x0 : x0 + target_w]

    # Finally, ensure multiples of tile_size (guard against rounding)
    final_h = _enforce_multiple(cropped.shape[0], tile_size)
    final_w = _enforce_multiple(cropped.shape[1], tile_size)
    cropped = cropped[:final_h, :final_w]
    return cropped

def resize_to_fixed(img_rgb, size=(32, 32)):
    return cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
