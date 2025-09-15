from __future__ import annotations
import numpy as np

def match_by_mean_rgb(
    patch_means: np.ndarray,  # (K, 3) float32
    tile_means: np.ndarray,   # (N, 3) float32
) -> np.ndarray:
    """
    Returns best tile index per patch via L2 on mean RGB. Vectorized.
    patch_means: Kx3, tile_means: Nx3 -> distances KxN (broadcast)
    """
    # (K,1,3) - (1,N,3) -> (K,N,3)
    diff = patch_means[:, None, :] - tile_means[None, :, :]
    d2 = (diff * diff).sum(axis=2)   # (K, N)
    return np.argmin(d2, axis=1)     # (K,)

def match_by_mse(
    patches_flat: np.ndarray,  # (K, D) float32 in [0,1]
    tiles_flat: np.ndarray,    # (N, D) float32 in [0,1]
    chunk: int = 1024,
) -> np.ndarray:
    """
    Returns best tile index per patch via pixel MSE. Chunked to bound memory for large K or N.
    """
    K, D = patches_flat.shape
    N, D2 = tiles_flat.shape
    assert D == D2

    out = np.empty((K,), dtype=np.int32)
    # Process in chunks over K (patches)
    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        P = patches_flat[s:e]  # (k, D)
        # (k,1,D) - (1,N,D) -> (k,N,D)
        diff = P[:, None, :] - tiles_flat[None, :, :]
        d2 = (diff * diff).mean(axis=2)  # (k, N)
        out[s:e] = np.argmin(d2, axis=1)
    return out
