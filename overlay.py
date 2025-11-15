import cv2
import numpy as np

GREEN = (0,255,0)  # Left
BLUE  = (255,0,0)  # Right
GRAY  = (200,200,200)

def _draw_line(img, pts, color, thickness=8, dashed=False):
    if pts is None: return
    x1,y1,x2,y2 = map(int, pts)
    if dashed:
        # simple dashed: draw small segments
        n = 16
        for i in range(n):
            t0 = i / n; t1 = (i+0.5) / n
            xa = int(x1 + (x2-x1)*t0); ya = int(y1 + (y2-y1)*t0)
            xb = int(x1 + (x2-x1)*t1); yb = int(y1 + (y2-y1)*t1)
            cv2.line(img, (xa,ya), (xb,yb), color, thickness)
    else:
        cv2.line(img, (x1,y1), (x2,y2), color, thickness)
        

def draw_overlay(frame, lanes, Minv=None):
    out = frame.copy()
    # Left: green when confident, dashed gray otherwise
    if lanes['left_pts'] is not None:
        _draw_line(out, lanes['left_pts'],      #use _draw_line function to go
                   GREEN if lanes['left_flag'] else GRAY,
                   thickness=8, dashed=(lanes['left_flag']==0))
    # Right: blue when confident, dashed gray otherwise
    if lanes['right_pts'] is not None:
        _draw_line(out, lanes['right_pts'],
                   BLUE if lanes['right_flag'] else GRAY,
                   thickness=8, dashed=(lanes['right_flag']==0))
    # HUD
    text = f"Left: {'YES' if lanes['left_flag'] else 'NO'} | Right: {'YES' if lanes['right_flag'] else 'NO'} | Conf: {0.5*(lanes['left_conf']+lanes['right_conf']):.2f}"
    cv2.putText(out, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return out

# def _draw_quadratic(img, fit, color, dashed=False):
#     if fit is None: return
#     h = img.shape[0]
#     ys = np.linspace(int(0.6*h), h-1, num=200)
#     xs = fit[0]*ys**2 + fit[1]*ys + fit[2]
#     pts = np.vstack([xs, ys]).T.astype(int)
#     cv2.polylines(img, [pts], False, color, 6)