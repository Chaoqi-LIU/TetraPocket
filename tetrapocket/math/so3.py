import pytorch3d.transforms
import torch
import pytorch3d
from typing import Optional, overload


def random_so3(
    n: int,
    representation: str = 'rotation_matrix',
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate random SO(3) rotations.
    :param n: number of random rotations
    :param representation: representation, str, default 'rotation_matrix'
    :param dtype: data type, torch.dtype
    :param device: device, torch.device
    :return: random SO(3) rotations in the specified representation
    """
    assert representation in ['rotation_matrix', 'quaternion', 'axis_angle', 'roll_pitch_yaw'], (
        f'invalid representation: {representation}'
        '\n(choose from "rotation_matrix", "quaternion", "axis_angle", "roll_pitch_yaw")'
    )
    quat = pytorch3d.transforms.random_quaternions(n, dtype=dtype, device=device)

    if representation == 'rotation_matrix':
        return quaternion_to_rotation_matrix(quat)
    elif representation == 'quaternion':
        return quat
    elif representation == 'axis_angle':
        return quaternion_to_axis_angle(quat)
    elif representation == 'roll_pitch_yaw':
        return quaternion_to_rollpitchyaw(quat)


"""
A series of functions to convert between different representations of 3D rotations. 
The directed graph of conversions is as follows:
    rpy <--> rotation matrix <--> quaternion <--> axis-angle
where a --> b means a can be converted to b without intermediate conversions.
"""

def rotation_matrix_to_quaternion(
    matrix: torch.Tensor   # (*, 3, 3)
) -> torch.Tensor:         # (*, 4)
    """
    Convert rotation matrix to quaternion.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: quaternion, (*, 4)
    """
    return pytorch3d.transforms.matrix_to_quaternion(matrix)


def rotation_matrix_to_axis_angle(
    matrix: torch.Tensor            # (*, 3, 3)
) -> torch.Tensor:                  # (*, 3)
    """
    Convert rotation matrix to axis-angle.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: axis-angle, (*, 3)
    """
    return pytorch3d.transforms.matrix_to_axis_angle(matrix)


def rotation_matrix_to_rollpitchyaw(
    matrix: torch.Tensor            # (*, 3, 3)
) -> torch.Tensor:                  # (*, 3)
    """
    Convert rotation matrix to roll-pitch-yaw.
    :param rotation_matrix: rotation matrix, (*, 3, 3)
    :return: roll-pitch-yaw, (*, 3)
    """
    return pytorch3d.transforms.matrix_to_euler_angles(matrix, 'XYZ')


def quaternion_to_axis_angle(
    quat: torch.Tensor      # (*, 4)
) -> torch.Tensor:          # (*, 3)
    """
    Convert quaternion to axis-angle.
    :param quat: quaternion, (*, 4)
    :return: axis-angle, (*, 3)
    """
    return pytorch3d.transforms.quaternion_to_axis_angle(quat)


def quaternion_to_rotation_matrix(
    quaternions: torch.Tensor     # (*, 4)
) -> torch.Tensor:                # (*, 3, 3)
    """
    Convert quaternion to rotation matrix.
    :param quaternions: quaternion, (*, 4)
    :return: rotation matrix, (*, 3, 3)
    """
    return pytorch3d.transforms.quaternion_to_matrix(quaternions)


def quaternion_to_rollpitchyaw(
    quaternions: torch.Tensor     # (*, 4)
) -> torch.Tensor:                # (*, 3)
    """
    Convert quaternion to roll-pitch-yaw.
    :param quaternions: quaternion, (*, 4)
    :return: roll-pitch-yaw, (*, 3)
    """
    assert quaternions.shape[-1] == 4, f'Invalid quaternion has shape: {quaternions.shape}'
    return rotation_matrix_to_rollpitchyaw(quaternion_to_rotation_matrix(quaternions))


def axis_angle_to_quaternion(
    axis_angle: torch.Tensor    # (*, 3)
) -> torch.Tensor:              # (*, 4)
    """
    Convert axis-angle to quaternion.
    :param axis_angle: axis-angle, (*, 3)
    :return: quaternion, (*, 4)
    """
    return pytorch3d.transforms.axis_angle_to_quaternion(axis_angle)


def axis_angle_to_rotation_matrix(
    axis_angle: torch.Tensor    # (*, 3)
) -> torch.Tensor:              # (*, 3, 3)
    """
    Convert axis-angle to rotation matrix.
    :param axis_angle: axis-angle, (*, 3)
    :return: rotation matrix, (*, 3, 3)
    """
    return pytorch3d.transforms.axis_angle_to_matrix(axis_angle)


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
    ], dim=-2)

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


class Rotation3d:
    """
    A class to represent SO(3) rotations.
    """
    def __init__(self, data: torch.Tensor, representation: str) -> None:
        """
        Initialize the Rotation3d class.
        :param data: rotation data, torch.Tensor
        :param representation: representation, str
        """
        assert representation in ['rotation_matrix', 'quaternion', 'axis_angle', 'roll_pitch_yaw'], (
            f'invalid representation: {representation}'
            '\n(choose from "rotation_matrix", "quaternion", "axis_angle", "roll_pitch_yaw")'
        )
        self.data_ = data
        self.representation_ = representation
        self.device_ = data.device
        self.dtype_ = data.dtype

    def __repr__(self) -> str:
        return f'Rotation3d(\n{self.data_}, \nrepresentation="{self.representation_}")'
    
    def data(self) -> torch.Tensor:
        """
        Get the rotation data.
        :return: rotation data, torch.Tensor
        """
        return self.data_
    
    def representation(self) -> str:
        """
        Get the representation.
        :return: representation, str
        """
        return self.representation_
    
    @overload
    def to(self, representation: str) -> 'Rotation3d':
        pass

    @overload
    def to(self, device: torch.device) -> 'Rotation3d':
        pass

    @overload
    def to(self, representation: str, device: torch.device) -> 'Rotation3d':
        pass

    def to(self, *args) -> 'Rotation3d':
        """
        Convert the rotation to another representation or device.
        :param representation: representation, str
        :param device: device, torch.device
        :return: converted rotation, Rotation3d

        If already in the target representation and device, return self.
        Otherwise, return the copy of the rotation in the target representation and device.
        """
        if len(args) == 1 and isinstance(args[0], str):
            representation = args[0]
            device = self.device_
        elif len(args) == 1 and isinstance(args[0], torch.device):
            representation = self.representation_
            device = args[0]
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], torch.device):
            representation = args[0]
            device = args[1]
        else:
            raise ValueError('invalid arguments')

        if device == self.device_ and representation == self.representation_:
            return self

        # data conversion
        if self.representation_ == 'rotation_matrix' and representation == 'quaternion':
            data = rotation_matrix_to_quaternion(self.data_)
        elif self.representation_ == 'rotation_matrix' and representation == 'axis_angle':
            data = rotation_matrix_to_axis_angle(self.data_)
        elif self.representation_ == 'rotation_matrix' and representation == 'roll_pitch_yaw':
            data = rotation_matrix_to_rollpitchyaw(self.data_)
        
        elif self.representation_ == 'quaternion' and representation == 'rotation_matrix':
            data = quaternion_to_rotation_matrix(self.data_)
        elif self.representation_ == 'quaternion' and representation == 'axis_angle':
            data = quaternion_to_axis_angle(self.data_)
        elif self.representation_ == 'quaternion' and representation == 'roll_pitch_yaw':
            data = quaternion_to_rollpitchyaw(self.data_)

        elif self.representation_ == 'axis_angle' and representation == 'rotation_matrix':
            data = axis_angle_to_rotation_matrix(self.data_)
        elif self.representation_ == 'axis_angle' and representation == 'quaternion':
            data = axis_angle_to_quaternion(self.data_)
        elif self.representation_ == 'axis_angle' and representation == 'roll_pitch_yaw':
            data = axis_angle_to_rollpitchyaw(self.data_)

        elif self.representation_ == 'roll_pitch_yaw' and representation == 'rotation_matrix':
            data = rollpitchyaw_to_rotation_matrix(self.data_)
        elif self.representation_ == 'roll_pitch_yaw' and representation == 'quaternion':
            data = rollpitchyaw_to_quaternion(self.data_)
        elif self.representation_ == 'roll_pitch_yaw' and representation == 'axis_angle':
            data = rollpitchyaw_to_axis_angle(self.data_)

        else:
            raise ValueError(f'cannot convert from {self.representation_} to {representation}')
        
        # to device
        data = data.to(device)

        return Rotation3d(data, representation)


def linear_so3_interpolation(
    start: torch.Tensor,                         # (*, ...)
    end: torch.Tensor,                           # (*, ...)
    num_segment: Optional[torch.Tensor] = None,  # (*,)
    timestamps: Optional[torch.Tensor] = None,   # (*, max_num_waypoints)
    representation: str = 'rotation_matrix',
    padding_value: float = float('nan'),
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
    angles: torch.Tensor = angles * timestamps
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