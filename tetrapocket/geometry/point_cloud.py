import scipy
import numpy as np
import torch
import pytorch3d
from typing import Optional, Union



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


def farthest_point_sampling(
    points: torch.Tensor, 
    K: Union[int, float, torch.Tensor],
    lengths: Optional[torch.Tensor] = None,
    random_start_point: bool = False,
    return_pts: bool = True,
    return_idx: bool = True,
) -> torch.Tensor:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a float, then the number of samples is 
            selected as a fraction of the total number of points in the point cloud.
            If K is a tensor, then it should have shape (N,) and with dtype either
            int or float.
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.
        return_pts: bool, if False, does not return the selected points.
        return_idx: bool, if False, does not return the indices of the selected points.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K). If return_pts is False, this
            array is not returned.
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K). If return_idx is False, this
            array is not returned.
    """
    assert not (return_pts is False and return_idx is False), \
        "At least one of return_pts or return_idx must be True."

    N, P, D = points.shape
    device = points.device

    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # standardize K (int64)
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, float):
        K = torch.floor(K * lengths.float()).to(torch.int64)
    elif torch.is_tensor(K):
        if K.shape != (N,):
            raise ValueError("points and K must have same batch dimension.")
        dtype = K.dtype
        if dtype == torch.int:
            pass
        elif dtype == torch.float:
            K = torch.floor(K * lengths.float()).to(torch.int64)
        else:
            raise ValueError("K.dtype must be either int or float.")
    else:
        raise ValueError("K must be either int, float, or tensor.")
    
    sampled_points, idx = pytorch3d.ops.sample_farthest_points(
        points=points,
        lengths=lengths,
        K=K,
        random_start_point=random_start_point,
    )

    if return_pts is False:
        return idx
    elif return_idx is False:
        return sampled_points
    else:
        return sampled_points, idx
