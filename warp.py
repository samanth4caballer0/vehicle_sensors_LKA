import cv2
import numpy as np

def warp_to_birdeye(binary_mask):
    h, w = binary_mask.shape[:2]
    M = np.eye(3, dtype=np.float32)
    Minv = np.eye(3, dtype=np.float32)
    return binary_mask, M, Minv
