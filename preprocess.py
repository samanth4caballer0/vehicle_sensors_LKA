import cv2
import numpy as np

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    h, w = image.shape[:2]
    triangle = np.array([[
        (int(0.18*w), h),
        (int(0.82*w), h),
        (int(0.5*w),  int(0.42*h))
    ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(image, mask)

def threshold_hls_sobel(img):
    # canny+ROI 
    edges = detect_edges(img)
    masked = region_of_interest(edges)
    return masked