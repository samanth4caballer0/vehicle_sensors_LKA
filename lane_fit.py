import cv2
import numpy as np

def _make_coordinates(image_h, image_w, slope, intercept):
    # Build a line segment from slope/intercept — use two y's.
    y1 = image_h
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 1e-6
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=int)

def _average_slope_intercept(image, lines):
    h, w = image.shape[:2]
    left_fit, right_fit = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        # fit y = m x + b  → but we need x from y later; keep m,b
        params = np.polyfit((x1, x2), (y1, y2), 1)
        m, b = params[0], params[1]
        if m < 0:
            left_fit.append((m, b))
        else:
            right_fit.append((m, b))

    left_line = right_line = None
    left_conf = right_conf = 0.0

    if len(left_fit) > 0:
        m, b = np.mean(left_fit, axis=0)
        left_line = _make_coordinates(h, w, m, b)
        # confidence: count and slope magnitude
        left_conf = min(1.0, 0.2 + 0.15*len(left_fit) + 0.3*min(1.0, abs(m)))

    if len(right_fit) > 0:
        m, b = np.mean(right_fit, axis=0)
        right_line = _make_coordinates(h, w, m, b)
        right_conf = min(1.0, 0.2 + 0.15*len(right_fit) + 0.3*min(1.0, abs(m)))

    return left_line, right_line, left_conf, right_conf

def sliding_window_fit(binary_mask, prior=None):
    """
    Hough-based lane extraction replacing sliding windows.
    Returns dict with 'left_pts'/'right_pts' (2-point polylines), confidences, flags, and lat_offset (approx).
    """
    h, w = binary_mask.shape[:2]
    lines = cv2.HoughLinesP(binary_mask, rho=2, theta=np.pi/180, threshold=100,
                            minLineLength=40, maxLineGap=5)

    left_line = right_line = None
    left_conf = right_conf = 0.0
    if lines is not None and len(lines) > 0:
        left_line, right_line, left_conf, right_conf = _average_slope_intercept(
            np.zeros((h, w, 3), dtype=np.uint8), lines)

    # Flags with τ≈0.6
    left_flag = int(left_conf > 0.6)
    right_flag = int(right_conf > 0.6)

    # Approx ego-lane center and lateral offset (pixels → meters heuristic: assume 3.7 m lane, map width portion)
    lane_center_x = None
    if left_line is not None and right_line is not None:
        xL = (left_line[0] + left_line[2]) // 2
        xR = (right_line[0] + right_line[2]) // 2
        lane_center_x = (xL + xR) / 2.0
    elif left_line is not None:
        lane_center_x = left_line[0] + 350  # fallback guess
    elif right_line is not None:
        lane_center_x = right_line[0] - 350

    cam_center_x = w / 2.0
    px_offset = 0.0 if lane_center_x is None else (lane_center_x - cam_center_x)
    # assume ~700 px ≈ 3.7 m → meters per px ~ 0.0053 (rough; just to fill CSV)
    lat_offset_m = float(px_offset * 0.0053)

    return dict(
        left_pts=left_line, right_pts=right_line,
        left_conf=left_conf, right_conf=right_conf,
        left_flag=left_flag, right_flag=right_flag,
        lat_offset=lat_offset_m
    )


