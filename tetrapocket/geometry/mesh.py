import open3d
import numpy as np
import os
from typing import (
    Optional,
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