import torch
import numpy as np



def rotation_matrix_to_quaternion(
    matrix: torch.Tensor   # (*, 3, 3)
) -> torch.Tensor:                  # (*, 4)
    """
    Convert rotation matrix to quaternion.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: quaternion, (*, 4)
    """
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)  # (B, 3, 3)
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

    if out.shape[0] == 1:   # if only one input matrix
        out = out.squeeze(0)

    return out


def quaternion_to_axis_angle(
    quat: torch.Tensor      # (*, 4)
) -> torch.Tensor:          # (*, 3)
    """
    Convert quaternion to axis-angle.
    :param quat: quaternion, (*, 4)
    :return: axis-angle, (*, 3)
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)  # (B, 4)
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

    if out.shape[0] == 1:   # if only one input quaternion
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

    if o.shape[0] == 1:  # if only one input quaternion
        o = o.squeeze(0)

    return o


def axis_angle_to_quaternion(
    axis_angle: torch.Tensor    # (*, 3)
) -> torch.Tensor:              # (*, 4)
    """
    Convert axis-angle to quaternion.
    :param axis_angle: axis-angle, (*, 3)
    :return: quaternion, (*, 4)
    """
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)  # (B, 3)
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
     
    if quaternions.shape[0] == 1:  # if only one input axis-angle
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


def rollpitchyaw_to_rotation_matrix(
    rollpitchyaw: torch.Tensor    # (*, 3)
) -> torch.Tensor:                # (*, 3, 3)
    """
    Convert roll-pitch-yaw to rotation matrix.
    :param rollpitchyaw: roll-pitch-yaw, (*, 3)
    :return: rotation matrix, (*, 3, 3)
    """
    if rollpitchyaw.dim() == 1:
        rollpitchyaw = rollpitchyaw.unsqueeze(0)  # (B, 3)
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
    ])

    if rotmat.shape[0] == 1:  # if only one input roll-pitch-yaw
        rotmat = rotmat.squeeze(0)

    return rotmat


def rotation_matrix_to_rollpitchyaw(
    matrix: torch.Tensor            # (*, 3, 3)
) -> torch.Tensor:                  # (*, 3)
    """
    Convert rotation matrix to roll-pitch-yaw.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: roll-pitch-yaw, (*, 3)
    """
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)  # (B, 3, 3)
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

    if matrix.shape[0] == 1:    # if only one input matrix
        rpy = rpy.squeeze(0)

    return rpy
