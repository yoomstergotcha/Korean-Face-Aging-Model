# geometry/offsets.py

def apply_offset(pts, idxs, dx=0.0, dy=0.0):
    """
    Apply (dx, dy) offset to selected landmark indices.
    """
    for i in idxs:
        pts[i][0] += dx
        pts[i][1] += dy
