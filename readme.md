# Interactive Image Mosaic Generator

Reconstruct any image with a grid of tiny tiles (mini-images). The app slices an upload into fixed-size cells (e.g., 32×32) and replaces each cell with the best-matching tile from a folder. Includes a Gradio UI with tabs for uploading/processing tiles, building mosaics, and evaluating quality/speed.

> Your folder may be named `image-mosaic` or `image-mosiac` (typo). Commands assume project root.

---

## ✨ Features

- Vectorized grid operations (NumPy reshape/block view; no Python loops).
- Tile matching via:
  - Mean RGB distance (fast), or
  - Patch-wise MSE (more detailed).
- Optional pre-classification to preserve local brightness/color:
  - Luma bins (intensity), or
  - RGB bins (e.g., 4×4×4 = 64 classes).
- Gradio UI tabs:
  - Tiles: Upload & Process (first tab; add tiles with optional fixed resize and quantization)
  - Mosaic Builder (Original | Grid | Mosaic)
  - Performance (MSE, SSIM, error heatmap, runtime)
- Script for preprocessing examples from the CLI.
- Tests for resize/crop and quantization.

---

## 🗂️ Structure

```
app/                      # Gradio UI
  app.py
  examples/               # originals uploaded via UI (tiles source)
src/                      # Library code
  preprocessing/          # resize & color quantization
  tiling/                 # grid ops, tile index, matchers
  mosaic/                 # compose (vectorized)
  metrics/                # similarity metrics
data/outputs/             # generated artifacts (gitignored)
  preprocessed/           # processed tiles written by UI / scripts
scripts/
  preprocess_examples.py  # CLI: fixed-size + quantize
tests/
reports/
requirements.txt
pyproject.toml            # optional (editable install)
```

---

## 🔧 Setup

Python **3.10–3.11** recommended.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional (recommended): make `src` importable everywhere
pip install -e .
```

Pinned deps (excerpt):

```
gradio==4.44.1
gradio_client==1.4.1
opencv-python==4.9.0.80
numpy==1.26.4
scikit-image==0.24.0
scikit-learn==1.5.1
Pillow==10.4.0
```

---

## 🚀 Quickstart

1) Launch the app

```bash
python app/app.py
# launches with share=True in this project for convenience
```

2) In the UI

- Go to the tab “Tiles: Upload & Process”.
  - Upload images to build your tile set.
  - Optional: set Fixed resize (e.g., 32 or 64). 0 keeps original.
  - Optional: enable Color quantization (kmeans or median_cut) with parameters.
  - Click “Process & Continue to Mosaic” → originals save to `app/examples`, processed tiles to `data/outputs/preprocessed`, and you’ll be switched to Mosaic Builder.
- In “Mosaic Builder”
  - Upload the image you want to reconstruct.
  - Tiles folder: point to `data/outputs/preprocessed` (default).
  - Choose Tile size (px), Matching metric, and Classifier (or none).
  - Click “Generate Mosaic”.
- In “Performance”
  - Run analysis to see MSE, SSIM, error heatmap, and runtime.

CLI alternative for preprocessing:

```bash
python -m scripts.preprocess_examples --fixed-size 32 --quant kmeans --kmeans-k 16
# writes: data/outputs/preprocessed/*_32x32(_quant_*).jpg
```

---

## 🧠 How it works

* **Grid slicing**: vectorized `block_view()` returns `(rows, cols, ts, ts, 3)`.
* **Features**:

  * Patches: per-cell **mean RGB** and/or flattened **\[0,1]** patch vector.
  * Tiles: cached **mean RGB** and flattened vectors.
* **Matching**:

  * **Global** (no classifier): argmin over all tiles.
  * **Classified**: assign both patches & tiles to bins; match within each bin.
* **Composition**: index lookup per cell, then reshape/transpose to stitch into the final mosaic (no loops).

Key modules:

* `src/tiling/grid.py` — grid ops & overlay
* `src/tiling/tile_index.py` — loads tiles, builds index
* `src/tiling/matchers.py` — mean-RGB & MSE matchers
* `src/mosaic/compose.py` — vectorized mosaic (+ classification)
* `src/mosaic/naive.py` — loop-based reference
* `src/metrics/similarity.py` — MSE, SSIM (and optional PSNR)
* `src/metrics/runtime.py` — timing & benchmark

---

## 📈 Performance & Quality

* **Metrics** (UI → *Performance* tab):

  * **MSE** (lower is better)
  * **SSIM** (0–1, higher is better)
  * **Error heatmap** (per-pixel MSE, JET colormap)
* **Runtime scaling** (UI → *Benchmark* tab or `scripts/benchmark_runtime.py`):

  * Cells ≈ `(H/ts) * (W/ts)` ⇒ halving `ts` → \~4× more cells ⇒ \~4× runtime
  * Vectorized ≪ naive for all settings
  * MSE matching > mean-RGB runtime (more dimensions)

---

## 🧪 Tests

```bash
pytest -q
```

Includes shape checks, range checks, and core pipeline sanity.

---

## 🧩 Troubleshooting

* **`ModuleNotFoundError: No module named 'src'`**

  * Run from project root **or** add at top of `app/app.py`:

    ```python
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    ```
  * Or `pip install -e .`.

* **Gradio OpenAPI schema error: `TypeError: argument of type 'bool' is not iterable`**

  * Upgrade: `pip install -U gradio gradio_client`
  * Or launch with `show_api=False` (already set in `app.py`).

* **“localhost is not accessible”**

  * App launches with `share=True`; keep it on if your network blocks localhost.

---

## 🧭 Mapping to Assignment Rubric

| Criterion    | Where it’s addressed                                      |
| ------------ | --------------------------------------------------------- |
| Correctness  | Vectorized grid ops, deterministic mapping (`compose.py`) |
| Creativity   | Classifiers (luma/RGB), swap tile packs, blend control    |
| Interface    | Gradio tabs (Builder / Performance / Benchmark)           |
| Performance  | MSE/SSIM + runtime + benchmark (vectorized vs. naive)     |
| Presentation | Grid overlay, error heatmap, report-ready CSV/figures     |

---

## 📝 Report Tips (`reports/performance_report.md`)

Include:

* Brief method overview (grid size, metric, classifier, blend).
* Qualitative results (side-by-sides).
* Quantitative: MSE/SSIM table & benchmark curves (ms vs. tile size).
* Discussion: runtime scales \~1/ts²; vectorized vs. naive gap; mean-RGB vs. MSE trade-offs.

---

## 🛳️ Deploy (optional)

* **Hugging Face Spaces**:

  * Keep exact versions in `requirements.txt`
  * Entrypoint: `app/app.py`
  * Add `Spacefile` if you want hardware/runtime hints
* **Notes**:

  * Avoid temporary share links for submission; Spaces is preferred.

---

## 👐 Contributing

PRs welcome for:

* New matchers (e.g., LAB/ΔE76 or feature-based)
* Smarter tile selection (diversity constraints, no immediate repeats)
* GPU acceleration (CuPy, PyTorch) for large tile sets

---

## 📜 License

MIT (or your course’s required license). Add a `LICENSE` file if needed.

---

## 🙌 Acknowledgments

Built for an image processing lab to blend **technical** rigor with **creative** output. Have fun experimenting!
