from pathlib import Path
from typing import Union
import cv2
import numpy as np

# cv2 loads BGR; convert to RGB to keep consistency across the codebase.
def load_image(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def save_image_rgb(path: Union[str, Path], img_rgb: np.ndarray, quality: int = 92) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    elif ext == ".png":
        cv2.imwrite(str(path), img_bgr)  # use default compression
    else:
        # fallback to PNG
        cv2.imwrite(str(path.with_suffix(".png")), img_bgr)

def list_images(folder: Union[str, Path]) -> list[Path]:
    folder = Path(folder)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
