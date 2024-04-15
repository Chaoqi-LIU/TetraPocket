import scipy
import numpy as np



def points_in_points_cvx_hull(
    points0: np.ndarray,    # (n0, d)
    points1: np.ndarray,    # (n1, d)
    tol: float = 1e-12    
) -> np.ndarray:            # (n0,)
    """
    Compute points in points convex hull.
    :param points0: points0, (n0, d)
    :param points1: points1, (n1, d)
    :param tol: tolerance, default 1e-12
    :return: boolean array, (n0,) indicating whether points0 in points1 convex hull

    Via experiments, among linear programming, scipy.spatial.Delaunay, etc. This method is 
    the fastest when both pts and pcd are large. details: https://stackoverflow.com/a/42165596
    """
    assert points0.shape[1] == points1.shape[1], 'dimension mismatch'
    cvx_hull = scipy.spatial.ConvexHull(points1)    # Ax + b ≤ 0
    return np.all(
        points0 @ cvx_hull.equations[:, :-1].T + cvx_hull.equations[:, -1] <= tol, axis=1
    )