# Unity is left-handed (y up); nestbox and the arm are right-handed. Mirroring x converts between them
# (it is its own inverse). Apply it at every boundary where Unity data enters or nestbox poses leave.
import numpy as np

MIRROR_X = np.diag([-1.0, 1.0, 1.0])

def mirror_point(p):
    p = np.asarray(p, dtype=float)
    return p * np.array([-1.0, 1.0, 1.0])

def mirror_covariance(cov):
    cov = np.asarray(cov, dtype=float)
    return MIRROR_X @ cov @ MIRROR_X

def mirror_quaternion(q):
    """Mirror a (w, x, y, z) rotation quaternion across the x axis: (w, x, -y, -z)."""
    w, x, y, z = q
    return [w, x, -y, -z]

def mirror_pose(origin, quaternion):
    """Convert a pose (origin, w-first quaternion) between Unity and nestbox handedness."""
    return mirror_point(origin).tolist(), mirror_quaternion(quaternion)
