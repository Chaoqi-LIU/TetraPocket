import sys
import os
import torch
import numpy as np
import scipy
import open3d
from typing import (
    Optional,
)



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
    

def interactive_pick_points_on_mesh_surface(
    mesh_path: str, 
    num_samples: int = 10000
) -> Optional[np.ndarray]:
    """
    Interactive pick points on mesh surface.
    :param mesh_path: mesh path
    :param num_samples: number of samples
    :return: points, (n, 3), in local frame

    Pick points by [shift + left click] on the mesh surface.
    """
    assert os.path.exists(mesh_path), f'mesh path {mesh_path} not exists'
    assert num_samples > 0, 'num_samples should be positive'
    mesh = open3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    viz = open3d.visualization.VisualizerWithEditing()
    viz.create_window()
    viz.add_geometry(pcd)
    viz.run()
    viz.destroy_window()
    picked_points_indices = np.asarray(viz.get_picked_points())
    if len(picked_points_indices) == 0:
        None
    else:
        return np.asarray(pcd.points)[picked_points_indices]