import os, csv, cv2
from preprocess import threshold_hls_sobel
from warp import warp_to_birdeye
from lane_fit import hough_line_fit
from temporal import temporal_smooth
from overlay import draw_overlay

VIDEO_PATH = os.environ.get("VIDEO", "data/night.mp4")
OUT_VIDEO  = "outputs/night4.mp4"
OUT_CSV    = "outputs/per_frame_night4.csv"

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)

def run():
    ensure_dirs()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    csv_f = open(OUT_CSV, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame_id","left_detected","right_detected","left_conf","right_conf","lat_offset_m"])

    prev = None
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = threshold_hls_sobel(frame)     # Canny + ROI
        bird, M, Minv = warp_to_birdeye(mask)

        lanes = hough_line_fit(bird, prior=prev)   # Hough variant
        lanes = temporal_smooth(lanes, prev=prev)

        overlay = draw_overlay(frame, lanes, Minv=None)
        writer.write(overlay)

        # Live preview window (press 'c' to quit early)
        cv2.imshow('result', overlay)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

        csv_w.writerow([fid, lanes['left_flag'], lanes['right_flag'],
                        f"{lanes['left_conf']:.3f}", f"{lanes['right_conf']:.3f}",
                        f"{lanes['lat_offset']:.4f}"])
        prev = lanes
        fid += 1

    cap.release(); writer.release(); csv_f.close()
    # Close preview windows
    cv2.destroyAllWindows()
    print("Done. See outputs")

if __name__ == "__main__":
    run()

# import os
# import csv
# import time
# import numpy as np
# import cv2

# from preprocess import threshold_hls_sobel, detect_edges, region_of_interest
# from warp import warp_to_birdeye
# from lane_fit import hough_line_fit
# from temporal import temporal_smooth
# from overlay import draw_overlay
# from metrics import write_pipeline_debug_video, compute_latency


# VIDEO_PATH = os.environ.get("VIDEO", "data/highway.mp4")
# OUT_VIDEO  = "outputs/debug.mp4"
# OUT_CSV    = "outputs/debug.csv"
# OUT_DEBUG  = "outputs/pipeline_debug.mp4"


# def ensure_dirs():
#     os.makedirs("outputs", exist_ok=True)


# def run():
#     ensure_dirs()
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     writer = cv2.VideoWriter(
#         OUT_VIDEO,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (w, h),
#     )

#     csv_f = open(OUT_CSV, "w", newline="")
#     csv_w = csv.writer(csv_f)
#     csv_w.writerow(
#         ["frame_id",
#          "left_detected", "right_detected",
#          "left_conf", "right_conf",
#          "lat_offset_m"]
#     )

#     # ---- for latency ----
#     frame_times = []

#     # ---- for debug mosaic video ----
#     frames_input      = []
#     frames_preproc    = []
#     frames_edges      = []
#     frames_roi        = []
#     frames_hough      = []
#     frames_classified = []

#     prev = None
#     fid = 0

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         t0 = time.time()   # start timing for this frame

#         # 0) input frame
#         frames_input.append(frame.copy())

#         # 1) preprocess (here: grayscale view just for visualization)
#         roi = threshold_hls_sobel(frame)   # returns single-channel mask (edges/ROI)
#         # roi may already be single-channel; if so use it directly as `gray`.
#         if roi is None:
#             gray = None
#         elif roi.ndim == 2:
#             gray = roi.copy()
#         else:
#             gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         frames_preproc.append(gray)

#         # 2) edges (Canny)
#         edges = detect_edges(roi)          # your blur + Canny
#         frames_edges.append(edges)

#         # 3) ROI mask applied to edges (triangle on road)
#         edges_roi = region_of_interest(edges)
#         frames_roi.append(edges_roi)

#         # 4) Hough transform (for detection AND for a debug image)
#         bird, M, Minv = warp_to_birdeye(edges_roi)  # identity for now
#         # Lane detection based on Hough (inside sliding_window_fit)
#         lanes = hough_line_fit(bird, prior=prev)

#         # build "Hough lines" visualization: raw lines drawn in red
#         # ensure a 3-channel image for drawing debug Hough lines
#         if roi is None:
#             hough_debug = np.zeros((h, w, 3), dtype=np.uint8)
#         elif roi.ndim == 2:
#             hough_debug = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
#         else:
#             hough_debug = roi.copy()
#         lines = cv2.HoughLinesP(
#             edges_roi, rho=2, theta=np.pi / 180,
#             threshold=100, minLineLength=40, maxLineGap=5
#         )
#         if lines is not None:
#             for x1, y1, x2, y2 in lines.reshape(-1, 4):
#                 cv2.line(hough_debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         frames_hough.append(hough_debug)

#         # 5) temporal smoothing + final classified lines overlay
#         lanes = temporal_smooth(lanes, prev=prev)
#         overlay = draw_overlay(frame, lanes, Minv=None)
#         frames_classified.append(overlay)

#         # write annotated frame to main output video
#         writer.write(overlay)

#         # write CSV row
#         csv_w.writerow([
#             fid,
#             lanes["left_flag"],
#             lanes["right_flag"],
#             f"{lanes['left_conf']:.3f}",
#             f"{lanes['right_conf']:.3f}",
#             f"{lanes['lat_offset']:.4f}",
#         ])

#         prev = lanes
#         fid += 1

#         # end timing for this frame
#         t1 = time.time()
#         frame_times.append(t1 - t0)

#     cap.release()
#     writer.release()
#     csv_f.close()

#     # ---- metrics: latency ----
#     latency_stats = compute_latency(frame_times)
#     print("Latency stats:", latency_stats)

#     # ---- debug mosaic video (2x3 panels) ----
#     #   [input | preprocess | edges]
#     #   [ROI   | Hough      | classify]
#     write_pipeline_debug_video(
#         out_path=OUT_DEBUG,
#         fps=fps,
#         frames_input=frames_input,
#         frames_preprocess=frames_preproc,
#         frames_edges=frames_edges,
#         frames_roi=frames_roi,
#         frames_hough=frames_hough,
#         frames_classified=frames_classified,
#     )

#     print("Done. See:")
#     print(f"output vid- {OUT_VIDEO}")
#     print(f"output csv- {OUT_CSV}")
#     print(f"debug vid- {OUT_DEBUG}")


# if __name__ == "__main__":
#     run()

