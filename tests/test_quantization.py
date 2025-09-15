import numpy as np
from src.preprocessing.color_quantization import quantize

def test_kmeans_runs():
    img = (np.random.rand(128, 128, 3) * 255).astype("uint8")
    out = quantize(img, method="kmeans", kmeans_k=8)
    assert out.shape == img.shape

def test_median_cut_runs():
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    out = quantize(img, method="median_cut", median_cut_colors=8)
    assert out.shape == img.shape
