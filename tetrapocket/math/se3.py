import torch
from typing import Optional

from tetrapocket.math import tensor, so3



def linear_se3_interpolation(
    start: torch.Tensor,                         # (*, 4, 4)
    end: torch.Tensor,                           # (*, 4, 4)
    num_segment: Optional[torch.Tensor] = None,  # (*,)
    timestamps: Optional[torch.Tensor] = None,   # (*, max_num_waypoints)
    padding_value: float = float('nan'),
) -> torch.Tensor:              # (*, max_num_waypoints, 4, 4)
    """
    Linear interpolation on SE(3).
    :param start: start, (*, 4, 4)
    :param end: end, (*, 4, 4)
    :param num_segment: number of segments, (*,)
    :param timestamps: timestamps, (*, max_num_waypoints)
    :param padding_value: padding value, default nan

    :return: interpolated waypoints, (*, max_num_waypoints, 4, 4)

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
    if start.dim() == 2:
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)
        unsqueezed = True
    batch_size = start.size(0)

    # SO3 interpolation
    Rt = so3.linear_so3_interpolation(
        start[:, :3, :3], end[:, :3, :3],
        num_segment=num_segment, timestamps=timestamps,
        representation='rotation_matrix',
    )

    # Rn interpolation
    pt = tensor.linear_tensor_interpolation(
        start[:, :3, 3], end[:, :3, 3],
        batched=True,
        num_segment=num_segment, timestamps=timestamps,
        padding_value=padding_value,
    )

    # concatenate
    Xt = torch.cat([
        torch.cat([Rt, pt[..., None]], dim=-1),
        torch.tensor([0.,0.,0.,1.], device=device).view(1,1,1,4)\
             .expand(batch_size, Rt.size(1), -1, -1)
    ], dim=-2)

    # padding with padding_value
    nan_mask = torch.isnan(Xt.view(batch_size, Xt.size(1), -1)).any(dim=-1)
    Xt[nan_mask] = padding_value

    if unsqueezed:
        Xt = Xt.squeeze(0)

    return Xt
