import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


# ============================================================
# 1) DETECTION ACCURACY
# ============================================================

def compute_detection_accuracy(gt_csv: str, pred_csv: str) -> Dict[str, float]:
    """
    Per-frame side detection accuracy.

    Assumes both CSVs have at least:
      frame_id,left_detected,right_detected

    gt_csv  : path to ground-truth CSV (TA labels or your annotations)
    pred_csv: path to your model's per_frame.csv

    Returns a dict with left, right and joint accuracy.
    """
    gt   = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)

    merged = gt.merge(pred,
                      on="frame_id",
                      suffixes=("_gt", "_pred"))

    left_correct  = (merged["left_detected_gt"]  == merged["left_detected_pred"]).mean()
    right_correct = (merged["right_detected_gt"] == merged["right_detected_pred"]).mean()
    both_correct  = ((merged["left_detected_gt"]  == merged["left_detected_pred"]) &
                     (merged["right_detected_gt"] == merged["right_detected_pred"])).mean()

    return {
        "left_accuracy":  float(left_correct),
        "right_accuracy": float(right_correct),
        "joint_accuracy": float(both_correct)
    }


# ============================================================
# 2) CURVE QUALITY
# ============================================================

def curve_mae_pixels(gt_pts: np.ndarray,
                     pred_pts: np.ndarray) -> float:
    """
    Mean absolute pixel error between a ground-truth lane centerline
    and a predicted centerline.

    Both gt_pts and pred_pts are Nx2 arrays of (x, y) pixel coordinates
    sampled at the same y-positions (or already re-sampled).
    """
    if gt_pts.shape != pred_pts.shape:
        raise ValueError("gt_pts and pred_pts must have the same shape (N,2).")

    diff = np.abs(gt_pts - pred_pts)
    # only care about lateral (x) error; you can also average both
    x_err = diff[:, 0]
    return float(np.mean(x_err))


def mask_iou(gt_mask: np.ndarray,
             pred_mask: np.ndarray) -> float:
    """
    IoU between lane masks.
    Inputs are binary masks (0 or 255, or 0/1).

    IoU = intersection / union.
    """
    # ensure boolean
    gt = gt_mask > 0
    pr = pred_mask > 0

    intersection = np.logical_and(gt, pr).sum()
    union        = np.logical_or(gt, pr).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


# ============================================================
# 3) TEMPORAL STABILITY
# ============================================================

def temporal_stability(lat_offsets: np.ndarray,
                       frame_ids: Optional[np.ndarray] = None) -> float:
    """
    Temporal stability of lateral offset on (ideally) straight segments.

    lat_offsets : 1D array of per-frame lateral offsets (in meters or pixels)
    frame_ids   : optional frame indices for bookkeeping (not used in calc)

    Returns standard deviation of lat_offsets.
    Lower std-dev = more stable (better).
    """
    lat_offsets = np.asarray(lat_offsets).astype(float)
    if lat_offsets.size == 0:
        return 0.0
    return float(np.std(lat_offsets))


# ============================================================
# 4) LATENCY
# ============================================================

def compute_latency(frame_times: List[float]) -> Dict[str, float]:
    """
    Compute latency stats from a list of per-frame processing times (in seconds).

    Typically you append (t_end - t_start) per frame in main.py,
    then pass that list here.

    Returns average, min, max latency and effective FPS.
    """
    if len(frame_times) == 0:
        return {"avg_latency_s": 0.0,
                "min_latency_s": 0.0,
                "max_latency_s": 0.0,
                "fps": 0.0}

    arr = np.array(frame_times, dtype=float)
    avg = arr.mean()
    return {
        "avg_latency_s": float(avg),
        "min_latency_s": float(arr.min()),
        "max_latency_s": float(arr.max()),
        "fps":           float(1.0 / avg if avg > 0 else 0.0)
    }


# ============================================================
# 5) MULTI-PANEL DEBUG VIDEO (like your screenshot)
# ============================================================

def _to_bgr(img: np.ndarray, size: tuple) -> np.ndarray:
    """
    Utility: resize to 'size' and ensure 3-channel BGR.
    """
    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    if img.ndim == 2:   # grayscale or edge map
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img_resized


def write_pipeline_debug_video(
    out_path: str,
    fps: float,
    frames_input:      List[np.ndarray],
    frames_preprocess: List[np.ndarray],
    frames_edges:      List[np.ndarray],
    frames_roi:        List[np.ndarray],
    frames_hough:      List[np.ndarray],
    frames_classified: List[np.ndarray],
) -> None:
    """
    Create a multi-panel video showing the pipeline stages, similar to:

        [ input        | preprocess | edges     ]
        [ roi_mask     | hough      | classify  ]

    All lists must have the same length (one entry per frame).
    Each entry is a numpy image (BGR or grayscale).

    out_path: path to output MP4
    fps     : frames per second for the output video
    """
    n_frames = len(frames_input)
    assert all(len(lst) == n_frames for lst in [
        frames_preprocess, frames_edges, frames_roi,
        frames_hough, frames_classified
    ]), "All frame lists must have the same length."

    if n_frames == 0:
        raise ValueError("No frames provided to write_pipeline_debug_video.")

    # Base size from the input frame
    h0, w0 = frames_input[0].shape[:2]
    tile_w, tile_h = w0, h0

    # Output video size: 2 rows x 3 cols
    out_w = tile_w * 3
    out_h = tile_h * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    for i in range(n_frames):
        img_in   = _to_bgr(frames_input[i],      (tile_w, tile_h))
        img_prep = _to_bgr(frames_preprocess[i], (tile_w, tile_h))
        img_edge = _to_bgr(frames_edges[i],      (tile_w, tile_h))
        img_roi  = _to_bgr(frames_roi[i],        (tile_w, tile_h))
        img_hgh  = _to_bgr(frames_hough[i],      (tile_w, tile_h))
        img_cls  = _to_bgr(frames_classified[i], (tile_w, tile_h))

        top_row    = np.hstack([img_in,  img_prep, img_edge])
        bottom_row = np.hstack([img_roi, img_hgh,  img_cls])
        mosaic     = np.vstack([top_row, bottom_row])

        writer.write(mosaic)

    writer.release()
    print(f"[metrics] Debug pipeline video saved to: {out_path}")
