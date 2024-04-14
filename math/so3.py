import torch
import numpy as np
from typing import (
    Optional,
    List,
    Tuple,
    Dict,
)


def random_quaternion(n: int) -> torch.Tensor:
    """
    Generate random quaternions.
    :param n: number of quaternions
    :return: quaternions, (n, 4)
    """
    def _copysign(a, b):
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)

    o = torch.randn(n, 4)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


"""
A series of functions to convert between different representations of 3D rotations. 
The directed graph of conversions is as follows:
    rpy <--> rotation matrix <--> quaternion <--> axis-angle
where a --> b means a can be converted to b without intermediate conversions.
"""

def rotation_matrix_to_quaternion(
    matrix: torch.Tensor   # (*, 3, 3)
) -> torch.Tensor:                  # (*, 4)
    """
    Convert rotation matrix to quaternion.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: quaternion, (*, 4)
    """
    unsqueezed = False
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)  # (B, 3, 3)
        unsqueezed = True
    assert matrix.shape[-2:] == (3, 3), \
        f'invalid rotation matrix has shape: {matrix.shape[-2:]}'

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1)

    def __sqrt_positive_part(x):
        # return torch.sqrt(torch.max(0, x)) but with a zero subgradient at x=0
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    q_abs = __sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ],dim=-1,))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack([
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
        #  `int`.
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
        #  `int`.
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
        #  `int`.
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
        #  `int`.
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2,)

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    out = torch.where(out[..., 0:1] < 0, -out, out)    

    if unsqueezed:
        out = out.squeeze(0)

    return out


def rotation_matrix_to_axis_angle(
    matrix: torch.Tensor            # (*, 3, 3)
) -> torch.Tensor:                  # (*, 3)
    """
    Convert rotation matrix to axis-angle.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: axis-angle, (*, 3)
    """
    assert matrix.shape[-2:] == (3, 3), f'invalid rotation matrix has shape: {matrix.shape[-2:]}'
    return quaternion_to_axis_angle(rotation_matrix_to_quaternion(matrix))


def rotation_matrix_to_rollpitchyaw(
    matrix: torch.Tensor            # (*, 3, 3)
) -> torch.Tensor:                  # (*, 3)
    """
    Convert rotation matrix to roll-pitch-yaw.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: roll-pitch-yaw, (*, 3)
    """
    unsqueezed = False
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)  # (B, 3, 3)
        unsqueezed = True
    assert matrix.shape[-2:] == (3, 3), f'invalid rotation matrix has shape: {matrix.shape[-2:]}'

    i0 = 0; i2 = 2
    tait_bryan = i0 != i2
    central_angle = torch.asin(
        matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
    )

    def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """
        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    rpy = torch.stack([
        _angle_from_tan("X", "Y", matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan("Z", "Y", matrix[..., i0, :], True, tait_bryan),
    ], dim=-1)

    if unsqueezed:
        rpy = rpy.squeeze(0)

    return rpy


def quaternion_to_axis_angle(
    quat: torch.Tensor      # (*, 4)
) -> torch.Tensor:          # (*, 3)
    """
    Convert quaternion to axis-angle.
    :param quat: quaternion, (*, 4)
    :return: axis-angle, (*, 3)
    """
    unsqueezed = False
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)  # (B, 4)
        unsqueezed = True
    assert quat.shape[-1] == 4, f'invalid quaternion has shape: {quat.shape[-1]}'

    norms = torch.norm(quat[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quat[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    out = quat[..., 1:] / sin_half_angles_over_angles

    if unsqueezed:
        out = out.squeeze(0)

    return out


def quaternion_to_rotation_matrix(
    quaternions: torch.Tensor     # (*, 4)
) -> torch.Tensor:                # (*, 3, 3)
    """
    Convert quaternion to rotation matrix.
    :param quaternions: quaternion, (*, 4)
    :return: rotation matrix, (*, 3, 3)
    """
    unsqueezed = False
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)  # (B, 4)

    assert quaternions.shape[-1] == 4, f'invalid quaternion has shape: {quaternions.shape[-1]}'

    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ), -1).reshape(quaternions.shape[:-1] + (3, 3))

    if unsqueezed:
        o = o.squeeze(0)

    return o


def quaternion_to_rollpitchyaw(
    quaternions: torch.Tensor     # (*, 4)
) -> torch.Tensor:                # (*, 3)
    """
    Convert quaternion to roll-pitch-yaw.
    :param quaternions: quaternion, (*, 4)
    :return: roll-pitch-yaw, (*, 3)
    """
    assert quaternions.shape[-1] == 4, f'invalid quaternion has shape: {quaternions.shape[-1]}'
    return rotation_matrix_to_rollpitchyaw(quaternion_to_rotation_matrix(quaternions))


def axis_angle_to_quaternion(
    axis_angle: torch.Tensor    # (*, 3)
) -> torch.Tensor:              # (*, 4)
    """
    Convert axis-angle to quaternion.
    :param axis_angle: axis-angle, (*, 3)
    :return: quaternion, (*, 4)
    """
    unsqueezed = False
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)  # (B, 3)
        unsqueezed = True
    assert axis_angle.shape[-1] == 3, f'invalid axis-angle has shape: {axis_angle.shape[-1]}'

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
     
    if unsqueezed:
        quaternions = quaternions.squeeze(0)

    return quaternions


def axis_angle_to_rotation_matrix(
    axis_angle: torch.Tensor    # (*, 3)
) -> torch.Tensor:              # (*, 3, 3)
    """
    Convert axis-angle to rotation matrix.
    :param axis_angle: axis-angle, (*, 3)
    :return: rotation matrix, (*, 3, 3)
    """
    assert axis_angle.shape[-1] == 3, f'invalid axis-angle has shape: {axis_angle.shape[-1]}'
    return quaternion_to_rotation_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_rollpitchyaw(
    axis_angle: torch.Tensor    # (*, 3)
) -> torch.Tensor:              # (*, 3)
    """
    Convert axis-angle to roll-pitch-yaw.
    :param axis_angle: axis-angle, (*, 3)
    :return: roll-pitch-yaw, (*, 3)
    """
    assert axis_angle.shape[-1] == 3, f'invalid axis-angle has shape: {axis_angle.shape[-1]}'
    return quaternion_to_rollpitchyaw(axis_angle_to_quaternion(axis_angle))


def rollpitchyaw_to_rotation_matrix(
    rollpitchyaw: torch.Tensor    # (*, 3)
) -> torch.Tensor:                # (*, 3, 3)
    """
    Convert roll-pitch-yaw to rotation matrix.
    :param rollpitchyaw: roll-pitch-yaw, (*, 3)
    :return: rotation matrix, (*, 3, 3)
    """
    unsqueezed = False
    if rollpitchyaw.dim() == 1:
        rollpitchyaw = rollpitchyaw.unsqueeze(0)  # (B, 3)
        unsqueezed = True
    assert rollpitchyaw.shape[-1] == 3, f'invalid roll-pitch-yaw has shape: {rollpitchyaw.shape[-1]}'

    roll, pitch, yaw = torch.unbind(rollpitchyaw, -1)

    rotmat = torch.stack([
        torch.stack([
            torch.cos(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll) - torch.sin(yaw) * torch.cos(roll),
            torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.sin(yaw) * torch.sin(roll),
        ], dim=-1),
        torch.stack([
            torch.sin(yaw) * torch.cos(pitch),
            torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.cos(roll),
            torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll) - torch.cos(yaw) * torch.sin(roll),
        ], dim=-1),
        torch.stack([
            -torch.sin(pitch),
            torch.cos(pitch) * torch.sin(roll),
            torch.cos(pitch) * torch.cos(roll),
        ], dim=-1),
    ], dim=1)

    if unsqueezed:
        rotmat = rotmat.squeeze(0)

    return rotmat


def rollpitchyaw_to_quaternion(
    rollpitchyaw: torch.Tensor    # (*, 3)
) -> torch.Tensor:                # (*, 4)
    """
    Convert roll-pitch-yaw to quaternion.
    :param rollpitchyaw: roll-pitch-yaw, (*, 3)
    :return: quaternion, (*, 4)
    """
    assert rollpitchyaw.shape[-1] == 3, f'invalid roll-pitch-yaw has shape: {rollpitchyaw.shape[-1]}'
    return rotation_matrix_to_quaternion(rollpitchyaw_to_rotation_matrix(rollpitchyaw))


def rollpitchyaw_to_axis_angle(
    rollpitchyaw: torch.Tensor    # (*, 3)
) -> torch.Tensor:                # (*, 3)
    """
    Convert roll-pitch-yaw to axis-angle.
    :param rollpitchyaw: roll-pitch-yaw, (*, 3)
    :return: axis-angle, (*, 3)
    """
    assert rollpitchyaw.shape[-1] == 3, f'invalid roll-pitch-yaw has shape: {rollpitchyaw.shape[-1]}'
    return quaternion_to_axis_angle(rollpitchyaw_to_quaternion(rollpitchyaw))


def linear_so3_interpolation(
    start: torch.Tensor,                         # (*, ...)
    end: torch.Tensor,                           # (*, ...)
    num_segment: Optional[torch.Tensor] = None,  # (*,)
    timestamps: Optional[torch.Tensor] = None,   # (*, max_num_waypoints)
    representation: Optional[str] = 'rotation_matrix',
    padding_value: Optional[float] = float('nan'),
) -> torch.Tensor:              # (*, max_num_waypoints, ...)
    """
    Linear interpolation on SO(3).
    :param start: start SO(3), (*, ...)
    :param end: end SO(3), (*, ...)
    :param num_segment: number of segments, (*,)
    :param timestamps: timestamps, (*, max_num_waypoints)
    :param representation: representation, str
    :param padding_value: padding value, float, default nan

    :return: interpolated SO(3), in its origin representation, (*, max_num_waypoints, ...)

    Either num_segment or timestamps should be provided. If both are provided, reports an error.
    """

    assert representation in ['rotation_matrix', 'quaternion', 'axis_angle', 'roll_pitch_yaw'], (
        f'invalid representation: {representation}'
        '\n(choose from "rotation_matrix", "quaternion", "axis_angle", "roll_pitch_yaw")'
    )
    assert start.dim() == end.dim(), 'start and end should have the same dimension'
    assert start.shape == end.shape, 'start and end should have the same shape'
    assert start.device == end.device == (num_segment.device if num_segment is not None else timestamps.device), \
        'start, end, num_segment, and timestamps should be on the same device'
    assert (num_segment is None) != (timestamps is None), 'either num_segment or timestamps should be provided'
    assert timestamps is None or (torch.isnan(timestamps) | ((timestamps >= 0) & (timestamps <= 1))).all(), \
        'timestamps should be in [0, 1] if not nan'
    
    device = start.device

    # add batch dimension if not present
    unsqueezed = False
    if (representation in ['rotation_matrix'] and start.dim() == 2) or \
       (representation in ['quaternion', 'axis_angle', 'roll_pitch_yaw'] and start.dim() == 1):
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)
        unsqueezed = True
    batch_size = start.shape[0]

    # all convert to rotation matrix
    if representation == 'quaternion':
        start = quaternion_to_rotation_matrix(start)
        end = quaternion_to_rotation_matrix(end)
    elif representation == 'axis_angle':
        start = axis_angle_to_rotation_matrix(start)
        end = axis_angle_to_rotation_matrix(end)
    elif representation == 'roll_pitch_yaw':
        start = rollpitchyaw_to_rotation_matrix(start)
        end = rollpitchyaw_to_rotation_matrix(end)

    # compute relative rotation
    rel_rot_mat = torch.bmm(torch.linalg.inv(start), end)   # (B, 3, 3)
    axes = rotation_matrix_to_axis_angle(rel_rot_mat)       # (B, 3)
    angles = torch.linalg.norm(axes, dim=-1, keepdim=True)  # (B, 1)
    zeros_mask = (torch.abs(angles) < 1e-8).squeeze(-1)     # (B,)
    axes[~zeros_mask] /= angles[~zeros_mask] 

    # set up timestamps if num_segment is provided
    if timestamps is None:
        num_waypoints = num_segment + 1                         # (B,)
        max_num_waypoints = num_waypoints.max().item()
        timestamps = torch.full((batch_size, max_num_waypoints), 
            fill_value=float('nan'), device=device)
        for i in range(batch_size):                             # TODO: can we do better?
            timestamps[i, :num_waypoints[i]] = \
                torch.linspace(0, 1, num_waypoints[i], device=device)
    else:
        max_num_waypoints = timestamps.shape[-1]

    # linear interpolation on axis-angle
    angles = angles * timestamps
    axes = axes[:, None, :].repeat(1, max_num_waypoints, 1)
    axes = axes * angles[..., None].expand(-1, -1, 3)
    rotmat = axis_angle_to_rotation_matrix(axes)
    rotmat = torch.einsum('bij,btjk->btik', start, rotmat)
    rotmat[zeros_mask] = start[zeros_mask, None, ...]

    # convert back to original representation
    if representation == 'quaternion':
        ret = rotation_matrix_to_quaternion(rotmat)
    elif representation == 'axis_angle':
        ret = rotation_matrix_to_axis_angle(rotmat)
    elif representation == 'roll_pitch_yaw':
        ret = rotation_matrix_to_rollpitchyaw(rotmat)
    else:
        ret = rotmat

    # padding with padding_value
    nan_mask = torch.isnan(timestamps.view(batch_size, max_num_waypoints, -1)).any(dim=-1)
    ret[nan_mask] = padding_value

    if unsqueezed:
        ret = ret.squeeze(0)

    return ret
