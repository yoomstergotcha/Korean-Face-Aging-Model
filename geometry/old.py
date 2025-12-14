# geometry/old.py

from .offsets import apply_offset
from . import landmarks as L

def apply_old(pts, strength=1.0):
    """
    Old-age geometry: gravity-dominant deformation
    """
    pts = pts.copy()
    s = strength

    # Brows
    apply_offset(pts, L.LEFT_BROW,  dx=+2*s, dy=+6*s)
    apply_offset(pts, L.RIGHT_BROW, dx=-2*s, dy=+6*s)

    # Eyes
    apply_offset(pts, L.UPPER_LID_L, dy=+10*s)
    apply_offset(pts, L.UPPER_LID_R, dy=+10*s)
    apply_offset(pts, L.LOWER_LID_L, dy=-4*s)
    apply_offset(pts, L.LOWER_LID_R, dy=-4*s)

    apply_offset(pts, L.LEFT_OUTER_CANTHUS,  dx=-3*s, dy=+12*s)
    apply_offset(pts, L.RIGHT_OUTER_CANTHUS, dx=+3*s, dy=+12*s)

    # Lips
    apply_offset(pts, L.UPPER_LIP_CENTER, dy=+4*s)
    apply_offset(pts, L.LOWER_LIP_CENTER, dy=+8*s)

    apply_offset(pts, L.LEFT_MOUTH_CORNER,  dx=-6*s, dy=+14*s)
    apply_offset(pts, L.RIGHT_MOUTH_CORNER, dx=+6*s, dy=+14*s)

    apply_offset(pts, L.UPPER_LIP_ARC, dy=+3*s)
    apply_offset(pts, L.LOWER_LIP_ARC, dy=+6*s)

    # Cheeks & jaw
    apply_offset(pts, L.LOWER_CHEEK_L, dx=-6*s, dy=+14*s)
    apply_offset(pts, L.LOWER_CHEEK_R, dx=+6*s, dy=+14*s)
    apply_offset(pts, L.JAWLINE, dy=+12*s)

    return pts
