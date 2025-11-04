import numpy as np

def temporal_smooth(curr, prev, alpha=0.8):
    if prev is None:
        return curr
    # Smooth confidences
    for k in ['left_conf', 'right_conf', 'lat_offset']:
        curr[k] = float(alpha*prev[k] + (1-alpha)*curr[k])
    # Smooth endpoints if both present
    for side in ['left_pts', 'right_pts']:
        if curr[side] is not None and prev[side] is not None:
            curr[side] = np.int32(alpha*prev[side] + (1-alpha)*curr[side])
    # Recompute flags after smoothing
    curr['left_flag']  = int(curr['left_conf']  > 0.6)
    curr['right_flag'] = int(curr['right_conf'] > 0.6)
    return curr
