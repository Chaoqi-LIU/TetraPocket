import torch
import numpy
from typing import (
    Optional,
)



def linear_Rn_interpolation(
    start: torch.Tensor,    # (*, d)
    end: torch.Tensor,      # (*, d)
    num_segment: Optional[torch.Tensor] = None,  # (*,)
    timestamps: Optional[torch.Tensor] = None,   # (*, max_num_waypoints)
    padding_value: Optional[float] = float('nan'),
) -> torch.Tensor:          # (*, max_num_waypoints, d)
    """
    Linear interpolation on R^n.
    :param start: start, (*, d)
    :param end: end, (*, d)
    :param num_segment: number of segments, (*,)
    :param timestamps: timestamps, (*, max_num_waypoints)
    :param padding_value: padding value, default nan

    :return: interpolated waypoints, (*, max_num_waypoints, d)

    Either num_segment or timestamps should be provided. If both are provided, reports an error.
    """
    assert (num_segment is None) != (timestamps is None), 'either num_segment or timestamps should be provided'
    assert start.device == end.device == (num_segment.device if num_segment is 
        not None else timestamps.device), 'device mismatch'
    assert start.dim() == end.dim(), 'dimension mismatch'
    assert start.shape == end.shape, 'shape mismatch'
    assert timestamps is None or (torch.isnan(timestamps) | ((timestamps >= 0) & (timestamps <= 1))).all(), \
        'timestamps should be in [0, 1] if not nan'

    device = start.device

    # add batch dimension
    unsqueezed = False
    if start.dim() == 1:
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)
        unsqueezed = True
    batch_size = start.size(0)

    # set up timestamps if num_segment is provided
    if num_segment is not None:
        num_waypoints = num_segment + 1
        max_num_waypoints = num_waypoints.max().item()
        print(f"batch_size: {batch_size}, max_num_waypoints: {max_num_waypoints}")
        timestamps = torch.full((batch_size, max_num_waypoints), 
            fill_value=float('nan'), device=device)
        for i in range(batch_size):                         # TOOD: can we do better?
            timestamps[i, :num_waypoints[i]] = \
                torch.linspace(0, 1, num_waypoints[i], device=device)
    else:
        max_num_waypoints = timestamps.size(1)

    # linear interpolation
    ret = torch.lerp(start[:, None, :], end[:, None, :], timestamps[:, :, None])

    # padding with padding_value
    nan_mask = torch.isnan(timestamps.view(batch_size, max_num_waypoints, -1)).any(dim=-1)
    ret[nan_mask] = padding_value
    
    if unsqueezed:
        ret = ret.squeeze(0)

    return ret
