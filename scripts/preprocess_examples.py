import argparse
from pathlib import Path
from src.config import (
    EXAMPLES_DIR, OUTPUTS_DIR, TILE_SIZE, GRID_COLS, GRID_ROWS, COLOR_QUANT_METHOD,
    KMEANS_K, MEDIAN_CUT_COLORS, DEFAULT_JPEG_QUALITY
)
from src.io_utils import list_images, load_image, save_image_rgb
from src.preprocessing.resize_crop import resize_and_center_crop_to_grid
from src.preprocessing.color_quantization import quantize
import cv2

def main():
    parser = argparse.ArgumentParser(description="Preprocess example images (Step 1).")
    parser.add_argument("--examples", type=str, default=str(EXAMPLES_DIR), help="Folder with input images")
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "preprocessed"), help="Output folder")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS if GRID_COLS else -1)
    parser.add_argument("--grid-rows", type=int, default=GRID_ROWS if GRID_ROWS else -1)
    parser.add_argument("--quant", type=str, default=COLOR_QUANT_METHOD, choices=["none","kmeans","median_cut"])
    parser.add_argument("--kmeans-k", type=int, default=KMEANS_K)
    parser.add_argument("--median-cut-colors", type=int, default=MEDIAN_CUT_COLORS)
    parser.add_argument("--fixed-size", type=int, default=None,
                        help="If set, directly resize to N×N (ignores grid/tile logic)")
    args = parser.parse_args()

    grid_cols = None if args.grid_cols == -1 else args.grid_cols
    grid_rows = None if args.grid_rows == -1 else args.grid_rows

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in list_images(args.examples):
        img = load_image(img_path)
        # If fixed-size is requested, do a direct resize then (optionally) quantize
        if args.fixed_size:
            stem = img_path.stem

            # 1) resize to N×N
            resized = cv2.resize(
                img, (args.fixed_size, args.fixed_size), interpolation=cv2.INTER_AREA
            )
            save_image_rgb(out_dir / f"{stem}_{args.fixed_size}x{args.fixed_size}.jpg", resized)
            print(f"✓ {img_path.name} -> resized to {args.fixed_size}x{args.fixed_size}")

            # 2) optional quantization on the resized image
            if args.quant and args.quant != "none":
                quant = quantize(
                    resized,
                    method=args.quant,
                    kmeans_k=args.kmeans_k,
                    median_cut_colors=args.median_cut_colors,
                )
                save_image_rgb(
                    out_dir / f"{stem}_{args.fixed_size}x{args.fixed_size}_quant_{args.quant}.jpg",
                    quant
                )
                print(f"  ↳ quantized with {args.quant}")
            continue


if __name__ == "__main__":
    main()
