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
