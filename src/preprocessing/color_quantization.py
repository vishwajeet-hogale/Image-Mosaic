from typing import Literal, Optional
import numpy as np
import cv2
from PIL import Image

QuantMethod = Literal["none", "kmeans", "median_cut"]

def quantize(
    img_rgb: np.ndarray,
    method: QuantMethod = "none",
    kmeans_k: int = 16,
    median_cut_colors: int = 16,
    kmeans_criteria_max_iter: int = 20,
    kmeans_attempts: int = 1,
) -> np.ndarray:
    """
    Apply optional color quantization.
    - "none": pass-through
    - "kmeans": OpenCV k-means in LAB for perceptual-ish distances
    - "median_cut": Pillow adaptive palette (median cut)
    """
    if method == "none":
        return img_rgb

    if method == "kmeans":
        # Convert to LAB for more perceptual clustering
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        Z = lab.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_criteria_max_iter, 1.0)
        compactness, labels, centers = cv2.kmeans(
            Z, kmeans_k, None, criteria, kmeans_attempts, cv2.KMEANS_PP_CENTERS
        )
        centers = centers.astype(np.uint8)
        quant_lab = centers[labels.flatten()].reshape(lab.shape)
        quant_rgb = cv2.cvtColor(quant_lab, cv2.COLOR_Lab2RGB)
        return quant_rgb

    if method == "median_cut":
        # PIL handles palette generation nicely
        pil = Image.fromarray(img_rgb)
        pal = pil.convert("P", palette=Image.ADAPTIVE, colors=int(median_cut_colors))
        out = pal.convert("RGB")
        return np.array(out)

    raise ValueError(f"Unknown quantization method: {method}")
