import torch
from typing import Optional



def uniform_sample(
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """
    Random uniform sampling.
    :param lb: lower bound
    :param ub: upper bound
    :return: with the same shape as lb and ub, float tensor
    """
    assert lb.device == ub.device, 'device mismatch'
    assert lb.shape == ub.shape, 'shape mismatch'
    return torch.rand(lb.shape, device=lb.device) * (ub - lb) + lb


def linear_tensor_interpolation(
    start: torch.Tensor,                                # (*, ...)
    end: torch.Tensor,                                  # (*, ...)
    batched: Optional[bool] = True,                     
    num_segment: Optional[torch.Tensor] = None,         # (*,)
    timestamps: Optional[torch.Tensor] = None,          # (*, max_num_waypoints)
    padding_value: Optional[float] = float('nan'),
) -> torch.Tensor:
    """
    A generalized linear interpolation for tensors with arbitrary dimensions / shapes.
    :param start: start tensor
    :param end: end tensor
    :param batched: whether the tensors are batched, if not, add batch dimension, 
                    but the output will not have batch dimension
    :param num_segment: number of segments
    :param timestamps: timestamps
    :param padding_value: padding value

    :return: interpolated tensor, (*, max_num_waypoints, ...)

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
    if not batched:
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)
    batch_size = start.size(0)

    # set up timestamps if num_segment is provided
    if num_segment is not None:
        num_waypoints = num_segment + 1
        max_num_waypoints = num_waypoints.max().item()
        timestamps = torch.full((batch_size, max_num_waypoints),
            fill_value=float('nan'), device=device)
        for i in range(batch_size):
            timestamps[i, :num_waypoints[i]] = \
                torch.linspace(0, 1, num_waypoints[i], device=device)
    else:
        max_num_waypoints = timestamps.size(1)

    # linear interpolation
    ret = torch.lerp(start[:, None, ...], end[:, None, ...], 
        timestamps.view(batch_size, max_num_waypoints, *([1] * (start.dim() - 1))))

    # padding with padding_value
    nan_mask = torch.isnan(timestamps.view(batch_size, max_num_waypoints, -1)).any(dim=-1)
    ret[nan_mask] = padding_value

    if not batched:
        ret = ret.squeeze(0)

    return ret
