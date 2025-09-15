from pathlib import Path

# Project roots
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "app" / "examples"
OUTPUTS_DIR = ROOT / "data" / "outputs"
CACHE_DIR = ROOT / "data" / "cache"
TILES_DIR = ROOT / "tiles"

# Preprocessing defaults
# Tile size in pixels (width == height). You can change in app later.
TILE_SIZE = 16

# Target grid (if set, we try to size image so width has this many tiles)
# You’ll typically set one of GRID_COLS or GRID_ROWS, not both.
GRID_COLS = 64
GRID_ROWS = None  # e.g., 48

# Quantization defaults
# Options: "none", "kmeans", "median_cut"
COLOR_QUANT_METHOD = "none"
KMEANS_K = 16  # palette size for k-means
MEDIAN_CUT_COLORS = 16

# JPEG/PNG default save params
DEFAULT_JPEG_QUALITY = 92

# Ensure dirs exist at import time (safe/no-op if present)
for _d in [OUTPUTS_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
