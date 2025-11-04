import cv2
import numpy as np

def crop_roi(img):
    # Keep full frame; we apply polygon mask instead (your triangle ROI)
    return img

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    h, w = image.shape[:2]
    # [(200,h),(1100,h),(550,250)]
    # Make it width-relative to avoid hard-coding 1100 for smaller videos
    triangle = np.array([[
        (int(0.18*w), h),
        (int(0.82*w), h),
        (int(0.5*w),  int(0.42*h))
    ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(image, mask)

def threshold_hls_sobel(img):
    # For compatibility with the baseline interface; we use Canny+ROI here.
    edges = detect_edges(img)
    masked = region_of_interest(edges)
    return masked


