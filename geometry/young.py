# geometry/young.py

from .offsets import apply_offset
from . import landmarks as L

def apply_young(pts, strength=1.0):
    """
    Young-age geometry: support-dominant deformation (smile + big eyes)
    """
    pts = pts.copy()
    s = 0.6 * strength  # young is weaker than old

    # Brows
    apply_offset(pts, L.LEFT_BROW,  dy=-6*s)
    apply_offset(pts, L.RIGHT_BROW, dy=-6*s)

    # Eyes (bigger)
    apply_offset(pts, L.UPPER_LID_L, dy=-6*s)
    apply_offset(pts, L.UPPER_LID_R, dy=-6*s)
    apply_offset(pts, L.LOWER_LID_L, dy=+4*s)
    apply_offset(pts, L.LOWER_LID_R, dy=+4*s)

    apply_offset(pts, L.LEFT_OUTER_CANTHUS,  dx=+2*s, dy=-8*s)
    apply_offset(pts, L.RIGHT_OUTER_CANTHUS, dx=-2*s, dy=-8*s)

    apply_offset(pts, L.LEFT_INNER_CANTHUS,  dx=-2*s)
    apply_offset(pts, L.RIGHT_INNER_CANTHUS, dx=+2*s)

    # Lips (smile)
    apply_offset(pts, L.LEFT_MOUTH_CORNER,  dx=+6*s, dy=-8*s)
    apply_offset(pts, L.RIGHT_MOUTH_CORNER, dx=-6*s, dy=-8*s)

    apply_offset(pts, L.UPPER_LIP_CENTER, dy=-3*s)
    apply_offset(pts, L.LOWER_LIP_CENTER, dy=-2*s)

    apply_offset(pts, L.UPPER_LIP_ARC, dy=-2*s)
    apply_offset(pts, L.LOWER_LIP_ARC, dy=-1*s)

    # Cheeks & jaw
    apply_offset(pts, L.LOWER_CHEEK_L, dx=+4*s, dy=-8*s)
    apply_offset(pts, L.LOWER_CHEEK_R, dx=-4*s, dy=-8*s)
    apply_offset(pts, L.JAWLINE, dy=-4*s)

    return pts
