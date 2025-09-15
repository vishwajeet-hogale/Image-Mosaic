# Interactive Image Mosaic Generator

Reconstruct any image with a grid of tiny **tiles** (mini-images). The app segments an upload into fixed-size cells (e.g., **32×32**) and replaces each cell with the **best-matching quantized tile**. Includes a Gradio UI with **Performance** and **Benchmark** tabs to evaluate quality (MSE/SSIM) and speed (vectorized vs. loop-based).

> Your folder may be named `image-mosaic` or `image-mosiac` (typo). Commands assume project root.

---

## ✨ Features

* **Vectorized grid ops** (NumPy reshape/block view — no Python loops).
* **Tile mapping** via:

  * Mean-RGB distance (fast), or
  * Patch-wise MSE (crisper but heavier).
* **Optional pre-classification** of patches & tiles:

  * **Luma bins** (intensity), or
  * **RGB bins** (e.g., 4×4×4 = 64 classes).
* **Gradio UI** with tabs:

  * **Mosaic Builder** (Original | Grid | Mosaic)
  * **Performance** (MSE, SSIM, error heatmap, runtime)
  * **Benchmark** (runtime vs. tile size; vectorized vs. naive)
* **Scripts** for preprocessing & benchmarking.
* **Tests** for grid, matchers, compose, metrics.

---

## 🗂️ Structure

```
app/                      # Gradio UI
  app.py
  examples/
src/                      # Library code
  preprocessing/          # resize & quantization
  tiling/                 # grid ops, tile index, matchers
  mosaic/                 # compose (vectorized) & naive reference
  metrics/                # similarity & runtime helpers
data/outputs/             # generated artifacts (gitignored)
  preprocessed/           # 32x32 quantized tiles from Step 1
  mosaics/
  benchmarks/
scripts/
  preprocess_examples.py  # Step 1
  benchmark_runtime.py    # Step 6
tests/
reports/
requirements.txt
pyproject.toml            # optional (for editable install)
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

## 🚀 Quickstart (end-to-end)

1. **Create a 32×32 quantized tile set** (Step 1)

```bash
python -m scripts.preprocess_examples --fixed-size 32 --quant kmeans --kmeans-k 16
# outputs: data/outputs/preprocessed/*_32x32_quant_kmeans.jpg
```

2. **Launch the app** (Steps 3–6 in UI)

```bash
python app/app.py
# the app is configured to use share=True (for networks that block localhost)
```

3. In the UI:

* **Upload** an image
* Choose **tile size** (default 32), **metric** (mean\_rgb|mse)
* Optionally enable **classifier** (luma\_bins|rgb\_bins)
* Adjust **blend** to mix back the original
* Use **Performance** tab for MSE/SSIM and heatmap
* Use **Benchmark** tab to compare vectorized vs. naive across tile sizes

---

## 📦 Scripts

### Step 1 — Preprocess & Quantize (examples → tiles)

```bash
python -m scripts.preprocess_examples --fixed-size 32 --quant kmeans --kmeans-k 16
# variations:
#   --quant none
#   --quant median_cut --median-cut-colors 16
```

### Step 6 — Runtime Benchmark (grid sizes & implementations)

```bash
python -m scripts.benchmark_runtime \
  --image app/examples/beach.jpg \
  --tiles data/outputs/preprocessed \
  --tile-sizes 16 32 64 \
  --metric mean_rgb \
  --repeats 3
```

Outputs CSV to `data/outputs/benchmarks/runtime_bench_mean_rgb.csv`.

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
