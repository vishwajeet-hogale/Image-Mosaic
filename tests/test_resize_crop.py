import numpy as np
from src.preprocessing.resize_crop import resize_and_center_crop_to_grid

def test_resize_multiple_of_tile():
    img = (np.random.rand(301, 517, 3) * 255).astype("uint8")
    out = resize_and_center_crop_to_grid(img, tile_size=16, grid_cols=32)
    h, w = out.shape[:2]
    assert h % 16 == 0 and w % 16 == 0
    assert w == 32 * 16  # 512 px width
