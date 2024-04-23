import torch
import numpy as np
from typing import (
    Optional,
)



def random_so2(
    n: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate random SO(2) rotations.
    :param n: Number of rotations to generate.
    :param device: Device to put the output tensor.
    :return: Random SO(2) rotations, in rotation matrix form, (n, 2, 2)
    """
    angles = torch.rand(n) * 2 * np.pi  # Uniformly sample angles in [0, 2pi).
    return torch.stack([
        torch.stack([torch.cos(angles), -torch.sin(angles)], dim=-1),
        torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1),
    ], dim=-1).to(device=device)
