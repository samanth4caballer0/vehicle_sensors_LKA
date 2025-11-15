
# RESULTS

DWKLEWFJIWWFHIW

## Night performance

![Night demo](outputs/night.gif)

## Highway day performance

![Highway day demo](outputs/highway.gif)

LALALALLAb

- RUN THE FOLLOWING COMMAND:
# Lane Keep Assist (LKA) — Python Video Lane Detection & Annotation

A compact, classical computer-vision lane detector implemented in Python with OpenCV and NumPy. The pipeline reads a driving video, detects left/right lane boundaries using Canny + Hough, fits quadratic polynomials, applies temporal smoothing, overlays the detected lanes on frames, and writes per-frame detection metrics to CSV. The repository also includes debug utilities that produce a 2x3 mosaic debug video showing intermediate pipeline stages.

## Table of Contents

- [Project Goals](#project-goals)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Outputs & Results](#outputs--results)
- [Design Choices & Limitations](#design-choices--limitations)
- [File Layout (what each file does)](#file-layout)
- [Development / Testing Tips](#development--testing-tips)

## Project Goals

This project provides a simple, interpretable lane-detection pipeline suitable for experiments and teaching. Goals:

- Detect left/right lane boundaries per frame and estimate a confidence for each side.
- Visualize detections by overlaying polylines and a small HUD on top of the original video.
- Produce an annotated output video and a CSV with per-frame detection metrics.
- Provide debug artifacts (mosaic video) to inspect intermediate stages.

## Pipeline Overview

High-level stages (implemented across `preprocess.py`, `warp.py`, `lane_fit.py`, `temporal.py`, `overlay.py`):

1. Preprocess / Thresholding — blur + Canny or HLS+Sobel depending on mode.
2. Region-of-interest (ROI) mask — keep the triangular road area.
3. Hough line detection — find line segments in the ROI.
4. Classify / group lines to left/right, build lane pixel sets.
5. Fit quadratic polynomials x(y) = a*y^2 + b*y + c for each side.
6. Temporal smoothing — smooth polynomial coefficients across frames.
7. Overlay — render solid/dashed polylines and a small HUD showing detection and confidence.

Diagram (conceptual):

```text
Input → Preprocess → Edges → ROI → Hough → Classify → Fit → Temporal → Overlay → Output
```

## Quick Start

1. Install dependencies (recommended in a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimum `requirements.txt`:

```
opencv-python
numpy
pandas
```

2. Run the main pipeline on the example video (press `c` to quit preview early):

```bash
VIDEO=data/highway.mp4 python main.py
```

Environment variables you can use:

- `VIDEO`: path to input video (defaults to `data/highway.mp4`).

The script produces: `outputs/annotated.mp4`, `outputs/per_frame_annotated.csv`, and `outputs/debug_annotated.mp4`.

## Outputs & Results

- `outputs/annotated.mp4` — final video with overlaid lane polylines and HUD.
- `outputs/per_frame_annotated.csv` — per-frame rows: `frame_id,left_detected,right_detected,left_conf,right_conf,lat_offset_m`.
- `outputs/debug_annotated.mp4` — 2×3 mosaic debug video showing: input, preprocess, edges, ROI, Hough lines debug, and final classified overlay.

Examples / GIFs: if you add `.gif` previews under `assets/` or `outputs/` you can embed them into this README using Markdown images, e.g. `![Night demo](outputs/night.gif)`.

## Design Choices & Limitations

- Approach: classical CV (Canny + Hough + polynomial fit). This is lightweight and interpretable but less robust than learning-based methods in challenging conditions (rain, heavy glare, worn lane paint).
- Performance: the pipeline is designed for simplicity rather than maximum speed. Real-time behavior depends on input resolution and CPU speed. The pipeline writes an annotated video and debug mosaic which may slow processing.
- Failure modes: misclassification of roadside edges as lanes, shaky polylines at night or low-contrast scenes. The temporal smoothing in `temporal.py` mitigates short dropouts.

## File Layout

- `main.py` — Entry point. Orchestrates reading frames, preprocessing, warping, lane detection, smoothing, overlay, and writes outputs (annotated video, CSV, debug video). Also shows a live preview window (press `c` to stop early).
- `preprocess.py` — Image preprocessing helpers: `detect_edges()` (blur + Canny), `region_of_interest()` (triangular mask), and `threshold_hls_sobel()` (an alternate HLS+Sobel path). This module returns single-channel masks used downstream.
- `warp.py` — (IPM) functions to warp to/from bird's-eye view. Used by the fitter when a top-down representation is helpful. In the baseline it's identity or simple homography utilities.
- `lane_fit.py` — Core detection/fitting logic. Runs `HoughLinesP` on edge masks, groups line segments per side, and fits quadratic polynomials x(y)=a*y^2+b*y+c. Returns per-side confidence and estimates lateral offset.
- `temporal.py` — Temporal smoothing utilities to stabilize polynomial coefficients and confidences across frames.
- `overlay.py` — Visualization: draws quadratic polylines, dashed fallbacks, and a small HUD showing detection flags and average confidence.
- `metrics.py` — Metrics and debug utilities: computes detection accuracy (with GT), mask IoU, latency stats, and writes the 2×3 mosaic debug video (`write_pipeline_debug_video`).
- `lane_detection.py` — Standalone demo script (single-file) that performs a simpler Hough-based lane detection flow and shows a live OpenCV preview (useful for interactive tinkering).
- `lf_ignore.py` — Legacy or alternative helper functions; may contain duplicate helpers used for experimentation.
- `data/` — Example input videos.
- `outputs/` — Auto-created results and debug artifacts (annotated videos, CSVs, generated GIFs if you make them).
- `calib/` — (optional) calibration data such as camera intrinsics or homography parameters used by `warp.py`.

## Development & Testing Tips

- To debug per-frame pipeline stages, `main.py` collects intermediate frames and writes the mosaic debug video `outputs/debug_annotated.mp4` (2×3 panels).
- If a GIF is not rendering in the README, ensure the GIF file is committed to the repository and referenced outside of a Markdown code block.
- If you run headless (no display), remove or guard calls to `cv2.imshow`/`cv2.destroyAllWindows()` or run with a virtual frame buffer (e.g., `xvfb-run`).
- To speed up experiments, downscale input frames (modify capture resolution or add resizing in `main.py`).

## Next steps you might want

- Add a `requirements.txt` / `pyproject.toml` and a small `Makefile` or `run.sh` wrapper for common experiments.
- Add unit tests for `metrics.py` to validate metric computations.
- Add a CLI entry (argparse) to `main.py` so you can toggle debug output, stride, and preview options.

---

If you'd like, I can also:

- Add a `requirements.txt` and commit it.
- Add a small CLI to `main.py` (`--no-preview`, `--stride`, `--out-dir`).
- Commit this README and push to `origin` for you.

Which of these would you like me to do next?

