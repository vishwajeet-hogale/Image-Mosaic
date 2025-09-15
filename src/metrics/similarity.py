# src/metrics/similarity.py
from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    return float(np.mean((a32 - b32) ** 2))

def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    # skimage >= 0.19 uses channel_axis instead of multichannel
    a_f = (a.astype(np.float32) / 255.0).clip(0, 1)
    b_f = (b.astype(np.float32) / 255.0).clip(0, 1)
    val = ssim(a_f, b_f, channel_axis=2, data_range=1.0, gaussian_weights=True, use_sample_covariance=False)
    return float(val)
