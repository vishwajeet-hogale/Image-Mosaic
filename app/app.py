# app/app.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import time
import numpy as np
import gradio as gr
import cv2

from src.tiling.grid import draw_grid_overlay
from src.tiling.tile_index import build_tile_index
from src.mosaic.compose import mosaic_from_uploaded
from src.metrics.similarity import mse as mse_metric, ssim_rgb as ssim_metric

DEFAULT_TILES_DIR = str(ROOT / "data" / "outputs" / "preprocessed")

# --------- helpers / cache ----------
_tile_db_cache = {}
def get_tile_db(tiles_dir: str, tile_size: int):
    key = (Path(tiles_dir).resolve(), int(tile_size))
    if key not in _tile_db_cache:
        _tile_db_cache[key] = build_tile_index(tiles_dir, size=tile_size)
    return _tile_db_cache[key]

def center_crop_to_shape(img: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Center-crop img to (H, W) if larger; assumes img is >= target size."""
    H, W = shape_hw
    h, w = img.shape[:2]
    y0 = max((h - H) // 2, 0)
    x0 = max((w - W) // 2, 0)
    return img[y0:y0+H, x0:x0+W]

def error_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel MSE heatmap (JET)."""
    diff2 = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=2)
    if float(diff2.max()) > 1e-8:
        norm = (255.0 * diff2 / diff2.max()).astype(np.uint8)
    else:
        norm = np.zeros_like(diff2, dtype=np.uint8)
    heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

# --------- core handlers ----------
def run_mosaic(
    image: np.ndarray,
    tiles_dir: str,
    tile_size: int,
    metric: str,
    classifier: str,
    luma_bins: int,
    rgb_bins: int,
    blend: float,
):
    if image is None:
        return None, None, None
    tile_db = get_tile_db(tiles_dir, tile_size)
    mosaic, _ = mosaic_from_uploaded(
        image,
        tile_db,
        tile_size=tile_size,
        metric=metric,                 # "mean_rgb" or "mse"
        classifier=classifier,         # "none", "luma_bins", "rgb_bins"
        luma_bins=luma_bins,
        rgb_bins=rgb_bins,
        blend=blend,
    )
    # Grid overlay for *uploaded* image (not cropped)
    grid = draw_grid_overlay(image, tile_size)
    # Return original, overlay, and mosaic
    return image, grid, mosaic

def run_performance(
    image: np.ndarray,
    tiles_dir: str,
    tile_size: int,
    metric: str,
    classifier: str,
    luma_bins: int,
    rgb_bins: int,
    blend: float,
):
    if image is None:
        return None, None, None, {"error": "Upload an image first."}

    tile_db = get_tile_db(tiles_dir, tile_size)
    t0 = time.perf_counter()
    mosaic, idx_map = mosaic_from_uploaded(
        image,
        tile_db,
        tile_size=tile_size,
        metric=metric,
        classifier=classifier,
        luma_bins=luma_bins,
        rgb_bins=rgb_bins,
        blend=blend,
    )
    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0

    # Compare vs cropped original (mosaic is built from a center-cropped region)
    H, W = mosaic.shape[:2]
    orig_crop = center_crop_to_shape(image, (H, W))

    # Metrics
    mse_val = mse_metric(orig_crop, mosaic)
    ssim_val = ssim_metric(orig_crop, mosaic)
    unique_tiles = int(np.unique(idx_map).size)
    cells = int(idx_map.size)

    metrics = {
        "tile_size": int(tile_size),
        "matching_metric": metric,
        "classifier": classifier,
        "luma_bins": int(luma_bins),
        "rgb_bins": int(rgb_bins),
        "blend": float(blend),
        "cells": cells,
        "unique_tiles_used": unique_tiles,
        "mse": mse_val,
        "ssim": ssim_val,
        "mapping_runtime_ms": round(runtime_ms, 2),
    }

    heat = error_heatmap(orig_crop, mosaic)
    return orig_crop, mosaic, heat, metrics

# --------- UI ----------
with gr.Blocks(title="Interactive Image Mosaic") as demo:
    gr.Markdown("## Image Mosaic — Build & Analyze\nA clearer layout with grouped controls and dynamic options.")

    def _toggle_bins(classifier_choice: str):
        show_luma = classifier_choice == "luma_bins"
        show_rgb = classifier_choice == "rgb_bins"
        return (
            gr.update(visible=show_luma),
            gr.update(visible=show_rgb),
        )

    with gr.Tabs():
        with gr.Tab("Mosaic Builder"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### Inputs")
                    in_img = gr.Image(type="numpy", label="Upload image")
                    tiles_dir = gr.Textbox(
                        value=DEFAULT_TILES_DIR,
                        label="Tiles folder",
                        info="Folder containing candidate tile images (.jpg/.png/...)"
                    )
                    gr.Markdown("### Controls")
                    tile_size = gr.Slider(8, 64, value=32, step=8, label="Tile size (px)")
                    metric = gr.Radio(
                        choices=["mean_rgb", "mse"], value="mean_rgb", label="Matching metric",
                        info="mean_rgb = faster, mse = more detailed but slower"
                    )
                    classifier = gr.Radio(
                        choices=["none", "luma_bins", "rgb_bins"], value="luma_bins", label="Classifier",
                        info="Partition patches and tiles before matching"
                    )
                    with gr.Row():
                        luma_bins = gr.Slider(2, 16, value=8, step=1, label="Luma bins", visible=True)
                        rgb_bins = gr.Slider(2, 6, value=4, step=1, label="RGB bins per channel", visible=False)
                    blend = gr.Slider(0.0, 0.6, value=0.0, step=0.05, label="Blend with original",
                                      info="0 = only tiles, higher = mix original image")
                    with gr.Accordion("How matching and classification work", open=False):
                        gr.Markdown(
                            """
                            - **Matching metric** decides how the best tile is selected for each image patch:
                              - **mean_rgb**: compares only the average color (fastest). Good for broad color mapping.
                              - **mse**: compares full per-pixel content (slower). Picks tiles that better match details.
                            - **Classifier** restricts which tiles are considered before matching:
                              - **none**: use all tiles for every patch.
                              - **luma_bins**: group patches/tiles by brightness; dark areas pick from dark tiles, bright from bright.
                              - **rgb_bins**: group by coarse RGB color bins; keeps color regions consistent.
                            - They work together: classification narrows candidates; the metric ranks the candidates to choose the winner.
                            """
                        )
                    run_btn = gr.Button("Generate Mosaic", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("### Outputs")
                    with gr.Row():
                        out1 = gr.Image(type="numpy", label="Original (uploaded)")
                        out2 = gr.Image(type="numpy", label="Grid Overlay")
                    out3 = gr.Image(type="numpy", label="Mosaic")

            classifier.change(
                fn=_toggle_bins,
                inputs=[classifier],
                outputs=[luma_bins, rgb_bins],
            )

            run_btn.click(
                fn=run_mosaic,
                inputs=[in_img, tiles_dir, tile_size, metric, classifier, luma_bins, rgb_bins, blend],
                outputs=[out1, out2, out3],
            )

        with gr.Tab("Performance"):
            gr.Markdown("### Evaluate mapping quality and speed")
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    in_img_p = gr.Image(type="numpy", label="Upload image")
                    tiles_dir_p = gr.Textbox(value=DEFAULT_TILES_DIR, label="Tiles folder")
                    tile_size_p = gr.Slider(8, 64, value=32, step=8, label="Tile size (px)")
                    metric_p = gr.Radio(choices=["mean_rgb", "mse"], value="mean_rgb", label="Matching metric")
                    classifier_p = gr.Radio(choices=["none", "luma_bins", "rgb_bins"], value="luma_bins", label="Classifier")
                    with gr.Row():
                        luma_bins_p = gr.Slider(2, 16, value=8, step=1, label="Luma bins", visible=True)
                        rgb_bins_p = gr.Slider(2, 6, value=4, step=1, label="RGB bins per channel", visible=False)
                    blend_p = gr.Slider(0.0, 0.6, value=0.0, step=0.05, label="Blend")
                    with gr.Accordion("How matching and classification work", open=False):
                        gr.Markdown(
                            """
                            - **Matching metric**:
                              - **mean_rgb**: average color matching (fast).
                              - **mse**: per-pixel comparison (slower, more detailed).
                            - **Classifier**:
                              - **none**: global tile pool.
                              - **luma_bins**: match within brightness bins.
                              - **rgb_bins**: match within coarse RGB bins.
                            - Flow: classify (optional) → match using selected metric.
                            """
                        )
                    run_perf = gr.Button("Run Performance Analysis")
                with gr.Column(scale=2):
                    with gr.Row():
                        orig_cropped = gr.Image(type="numpy", label="Original (center-cropped to mosaic)")
                        mosaic_img = gr.Image(type="numpy", label="Mosaic")
                    heat_img = gr.Image(type="numpy", label="Error Heatmap (per-pixel MSE)")
                    metrics_json = gr.JSON(label="Metrics (MSE, SSIM, runtime, usage)")

            def _toggle_bins_p(classifier_choice: str):
                return _toggle_bins(classifier_choice)

            classifier_p.change(
                fn=_toggle_bins_p,
                inputs=[classifier_p],
                outputs=[luma_bins_p, rgb_bins_p],
            )

            run_perf.click(
                fn=run_performance,
                inputs=[in_img_p, tiles_dir_p, tile_size_p, metric_p, classifier_p, luma_bins_p, rgb_bins_p, blend_p],
                outputs=[orig_cropped, mosaic_img, heat_img, metrics_json],
            )

if __name__ == "__main__":
    # Use share=True in your environment; hide API page to avoid schema bug
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False, debug=True)
