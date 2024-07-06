import scipy.interpolate
import torch
import scipy
import numpy as np
import torch_scatter
from typing import Optional, Tuple, Callable, Union, Sequence



def gaussian_smoothing(
    input: torch.Tensor,
    dim: int,
    kernel_size: Union[int, Sequence[int]],
    sigma: Union[float, Sequence[float]],
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
) -> torch.Tensor:
    """
    Apply gaussian smoothing to the input tensor.

    :param input: Input tensor, (B, C, *data_dim).
    :param dim: Dimension of the input data, only support 1D, 2D, 3D.
    :param kernel_size: Kernel size of the gaussian kernel.
    :param sigma: Standard deviation of the gaussian kernel.
    :param stride: Stride of the convolution.
    :param padding: Padding of the convolution.
    :return: Smoothed tensor.
    """

    dtype = input.dtype
    device = input.device
    C = input.size(1)

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, float):
        sigma = [sigma] * dim

    kernel: torch.Tensor = 1
    meshgrids = torch.meshgrid([
        torch.arange(size, dtype=dtype, device=device) 
        for size in kernel_size
    ], indexing='ij')
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mu = (size - 1) / 2
        kernel *= 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-((mgrid - mu) / std) ** 2 / 2)
    kernel = kernel / kernel.sum()

    # reshape kernel to depthwise conv
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(C, *[1] * (kernel.dim() - 1))

    if dim == 1:    conv = torch.nn.functional.conv1d
    elif dim == 2:  conv = torch.nn.functional.conv2d
    elif dim == 3:  conv = torch.nn.functional.conv3d
    else:           raise ValueError(f"Unsupported dim: {dim}")
    
    return conv(input, weight=kernel, groups=C, stride=stride, padding=padding)



def interpolate_2d_map_(
    maps: torch.Tensor,
    masks: torch.Tensor,
    method: str = 'linear',
) -> torch.Tensor:
    """
    Interpolate grid values for grids in `mask`, inplace version.

    :param maps: Grid values, (..., X, Y).
    :param masks: Mask, (..., X, Y), True if the grid needs to be interpolated.
    :param method: Interpolation method, 'linear' or 'nearest' or 'cubic'.
    :return: Interpolated map, (..., X, Y).
    """
    dtype = maps.dtype
    device = maps.device
    X, Y = maps.shape[-2:]
    pre_dims = maps.shape[:-2]
    maps = maps.view(-1, X, Y)
    masks = masks.view(-1, X, Y)

    for i in range(pre_dims.numel()):
        valid_grid = ~masks[i]
        maps[i, ~valid_grid] = torch.from_numpy(
            scipy.interpolate.griddata(
                torch.nonzero(valid_grid).to(dtype=dtype).cpu().numpy(),
                maps[i, valid_grid].cpu().numpy(),
                torch.nonzero(~valid_grid).to(dtype=dtype).cpu().numpy(),
                method=method
            )
        ).to(dtype=dtype, device=device)

    return maps.view(*pre_dims, X, Y)



def interpolate_2d_map(
    maps: torch.Tensor,
    masks: torch.Tensor,
    method: str = 'linear',
) -> torch.Tensor:
    """
    Interpolate grid values for grids in `mask`, return a new tensor.

    :param maps: Grid values, (..., X, Y).
    :param masks: Mask, (..., X, Y), True if the grid needs to be interpolated.
    :param method: Interpolation method, 'linear' or 'nearest' or 'cubic'.
    :return: Interpolated map, (..., X, Y).
    """
    dtype = maps.dtype
    device = maps.device
    X, Y = maps.shape[-2:]
    pre_dims = maps.shape[:-2]
    new_maps = maps.clone().view(-1, X, Y)
    masks = masks.view(-1, X, Y)

    for i in range(pre_dims.numel()):
        valid_grid = ~masks[i]
        new_maps[i, ~valid_grid] = torch.from_numpy(
            scipy.interpolate.griddata(
                torch.nonzero(valid_grid).to(dtype=dtype).cpu().numpy(),
                maps[i, valid_grid].cpu().numpy(),
                torch.nonzero(~valid_grid).to(dtype=dtype).cpu().numpy(),
                method=method
            )
        ).to(dtype=dtype, device=device)

    return new_maps.view(*pre_dims, X, Y)



def scatter_3d_points(
    points: torch.Tensor,
    to_index: Callable,
    map_size: Tuple[int, int],
    mask: torch.Tensor = None,
    padding: float = float('nan'),
    reduce: str = 'max',
) -> torch.Tensor:
    """
    Scatter 3D points to a 2D grid.

    :param points: Points, (..., N, 3), N is the number of points.
    :param to_index: Function to convert points to indices in the grid, if
           the point is out of bound, will be ignored.
    :param map_size: Size of the grid, (X, Y).
    :param mask: Mask, (..., N), 1 if the point is valid, 0 otherwise.
    :param padding: Padding value, if the grid is not filled.
    :param reduce: Reduce method, one of 'max', 'mean', 'std', 'sum', 'min', 
           'mul', 'logsumexp', 'softmax', 'log_softmax', 'count'.
    :return: Grid, (..., X, Y).
    """
    dtype = points.dtype
    device = points.device
    pre_dims = points.shape[:-2]
    X, Y = map_size

    # def scatter function
    if reduce == 'max':             scatter_fn = torch_scatter.scatter_max
    elif reduce == 'mean':          scatter_fn = torch_scatter.scatter_mean
    elif reduce == 'std':           scatter_fn = torch_scatter.scatter_std
    elif reduce == 'sum':           scatter_fn = torch_scatter.scatter_sum
    elif reduce == 'min':           scatter_fn = torch_scatter.scatter_min
    elif reduce == 'mul':           scatter_fn = torch_scatter.scatter_mul
    elif reduce == 'logsumexp':     scatter_fn = torch_scatter.scatter_logsumexp
    elif reduce == 'softmax':       scatter_fn = torch_scatter.scatter_softmax
    elif reduce == 'log_softmax':   scatter_fn = torch_scatter.scatter_log_softmax
    elif reduce == 'count':         pass
    else:                           
        raise ValueError(f"Unsupported reduce method: {reduce}, check with `help`")

    # indexing
    i, j = to_index(points)
    valid_mask = (i >= 0) & (i < X) & (j >= 0) & (j < Y) & (mask if mask is not None else 1)
    idx = i * Y + j
    idx += (torch.arange(torch.prod(torch.tensor(pre_dims)), device=device) * X * Y).view(*pre_dims, 1, 1)

    # count map has special treatment
    if reduce == 'count':
        maps = torch.zeros((*pre_dims, X, Y), dtype=dtype, device=device)
        maps.view(-1).index_add_(0, idx[valid_mask], torch.ones_like(idx[valid_mask]))
        return maps

    # other reduce methods
    maps = torch.full((*pre_dims, X, Y), float('nan'), dtype=dtype, device=device)
    z_val, z_arg = scatter_fn(points[valid_mask][..., 2], idx[valid_mask])
    to_remove = z_arg == idx[valid_mask].size(-1)
    z_val = z_val[~to_remove]
    z_arg = z_arg[~to_remove]
    maps.view(-1)[idx[valid_mask][z_arg]] = z_val

    # padding
    maps[torch.isnan(maps)] = padding

    return maps



def bilinear_interpolation(
    grid: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """
    Bilinear interpolation on 2D grid.
    :param grid: Grid, (N, X, Y).
    :param i: i indices, (N, P).
    :param j: j indices, (N, P).
    :param u: u residues, (N, P).
    :param v: v residues, (N, P).
    :return: Interpolated values, (N, P).
    """
    assert grid.ndim == 3, 'grid should have shape (N, X, Y)'
    N, X, Y = grid.shape
    assert i.shape == j.shape == u.shape == v.shape and \
        i.ndim == j.ndim == u.ndim == v.ndim == 2 and \
        i.shape[0] == N, 'i, j, u, v should have shape (N, P)'
    assert i.dtype == j.dtype == torch.long, \
        f'i, j should be torch.long, but received {i.dtype}, {j.dtype}'
    assert (i >= 0).all() and (i < X).all(), 'i should be in [0, X)'
    assert (j >= 0).all() and (j < Y).all(), 'j should be in [0, Y)'
    assert (u >= 0).all() and (u < 1).all(), 'u should be in [0, 1)'
    assert (v >= 0).all() and (v < 1).all(), 'v should be in [0, 1)'

    v00 = grid[:, i, j]
    v01 = grid[:, i, j + 1]
    v10 = grid[:, i + 1, j]
    v11 = grid[:, i + 1, j + 1]
    w0 = v00 + u * (v10 - v00)
    w1 = v10 + u * (v11 - v01)
    return w0 + v * (w1 - w0)



class Heightmap:
    def __init__(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        heights: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        2D heightmap representation. Currently, only support heightmaps with the same
        resolution, and the same size, i.e., heights[b].shape is the same for all b.

        :param frames SE(3), the center of the heightmap, (B, 4, 4).
        :param bboxes Bounding box of the heightmap, (B, 2, 2). [lb_x, lb_y; ub_x, ub_y], 
               expressed in local frame.
        :param heights Height values, (B, X, Y). heights[b, x, y] is the 
               z-value at (lb_x + x * dx, lb_y + y * dy).
        :param device Device to use for computation and output.
        """
        self.B_ = frames.shape[0]
        self.frames_W_L_ = frames
        self.bboxes_ = bboxes
        self.heights_ = heights
        self.device_ = device

        # infer
        self.frames_L_W_ = torch.linalg.inv(frames)
        self.X_: int = heights.shape[1]
        self.Y_: int = heights.shape[2]
        self.dimensions_: Tuple[torch.Tensor, torch.Tensor] = (
            bboxes[:, 1, 0] - bboxes[:, 0, 0],    # ub_x - lb_x
            bboxes[:, 1, 1] - bboxes[:, 0, 1]     # ub_y - lb_y
        )
        self.dx_ = self.dimensions_[0] / (self.X_ - 1)      # TODO: do we need to minus 1?
        self.dy_ = self.dimensions_[1] / (self.Y_ - 1)

    
    def to(self, device: torch.device) -> 'Heightmap':
        """
        Move the heightmap to another device.
        :param device: Device to move to.
        :return: Heightmap.
        """
        if self.device_ == device:
            return self
        return Heightmap(
            self.frames_W_L_.to(device),
            self.bboxes_.to(device),
            self.heights_.to(device),
            device=device
        )
    
    
    def indices_of(
        self,
        points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frames: str = 'world',
        return_residue: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return the indices of the points in the heightmap.
        :param points: Points, (N, P, 2 / 3)
        :param mask: Mask, (B,), 1 if the b-th heightmap is needed, 0 otherwise.
        :param frames: Frames of the points, 'world' or 'local'.
        :param return_residue: Return the residue or not.
        :return: Indices, ((i,j), (u,v)) -> heightmap[i, j] + residual (u * dx, v * dy).
                 (u, v) is returned only if return_residue is True.
        """
        assert points.ndim == 3 and (points.shape[-1] == 2 or points.shape[-1] == 3), \
            'points should have shape (N, P, 2) or (N, P, 3)'
        assert mask is None or (mask.ndim == 1 and mask.shape[0] == self.B_), 'mask should have shape (B,)'
        assert (mask is None and self.B_ == points.shape[0]) or \
            (mask is not None and mask.sum() == points.shape[0]), \
            '1s in mask should be equal to the number of points batch'
        assert frames in ['world', 'local'], 'frames should be either world or local'
        
        if mask is None:
            mask = torch.ones(self.B_, device=self.device_, dtype=torch.bool)
        
        points_L = points if frames == 'local' else self.world2local(points, mask=mask)

        u = (self.X_ - 1) * (points_L[:, :, 0] / self.dimensions_[0][mask][:, None] + 0.5)
        v = (self.Y_ - 1) * (points_L[:, :, 1] / self.dimensions_[1][mask][:, None] + 0.5)
        i = torch.floor(u).to(torch.long)
        j = torch.floor(v).to(torch.long)

        assert (i >= 0).all() and (i < self.X_).all(), 'i should be in [0, X), query points out of bound'
        assert (j >= 0).all() and (j < self.Y_).all(), 'j should be in [0, Y), query points out of bound'

        return (
            torch.stack([i, j], dim=-1),
            torch.stack([u - i, v - j], dim=-1) if return_residue else None
        )

    
    def height_at(
        self, 
        points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frames: str = 'world',
        interpolation: str = 'bilinear'
    ) -> torch.Tensor:
        """
        Query the height value at the points.
        :param points: Points, (N, P, 2 / 3).
        :param mask: Mask, (B,), 1 if the b-th heightmap is needed, 0 otherwise.
        :param frames: Frames of the points, 'world' or 'local'.
        :param interpolation: Interpolation method, 'bilinear' or 'nearest'.
        """
        assert points.ndim == 3 and (points.shape[-1] == 2 or points.shape[-1] == 3), \
            'points should have shape (N, P, 2) or (N, P, 3)'
        assert mask is None or (mask.ndim == 1 and mask.shape[0] == self.B_), 'mask should have shape (B,)'
        assert (mask is None and self.B_ == points.shape[0]) or \
            (mask is not None and mask.sum() == points.shape[0]), \
            '1s in mask should be equal to the number of points batch'
        assert frames in ['world', 'local'], 'frames should be either world or local'
        assert interpolation in ['bilinear', 'nearest'], 'interpolation should be either bilinear or nearest'

        indices, residue = self.indices_of(points, mask=mask, frames=frames, return_residue=True)
        i, j = torch.unbind(indices, dim=-1)
        u, v = torch.unbind(residue, dim=-1)

        if interpolation == 'bilinear':
            return bilinear_interpolation(self.heights_[mask], i, j, u, v)
        else:
            i += (u > 0.5).to(torch.long)
            j += (v > 0.5).to(torch.long)
            i = torch.clamp(i, 0, self.X_ - 1)  # TODO: maybe return None instead of clamp?
            j = torch.clamp(j, 0, self.Y_ - 1)
            return self.heights_[mask][:, i, j]


    def local2world(
        self,
        p_L: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert local points to world points.
        :param p_L: Local points, (N, P, 2 / 3).
        :param mask: Mask, (B,), 1 if the b-th heightmap is needed, 0 otherwise.
        :return: World points, (N, P, 2 / 3).
        """
        assert p_L.ndim == 3 and (p_L.shape[-1] == 2 or p_L.shape[-1] == 3), \
            'p_L should have shape (N, P, 3) or (N, P, 2)'
        N, P, _ = p_L.shape
        assert mask is None or (mask.ndim == 1 and mask.shape[0] == self.B_), \
            'mask should have shape (B,)'
        assert (mask is None and self.B_ == p_L.shape[0]) or \
            (mask is not None and mask.sum() == p_L.shape[0]), \
            '1s in mask should be equal to the number of points batch'
        
        d = p_L.shape[-1]
        if d == 2:  # padding to 3D.
            p_L = torch.cat([p_L, torch.zeros(N, P, 1, device=self.device_)], dim=-1)

        if mask is None:
            mask = torch.ones(self.B_, device=self.device_, dtype=torch.bool)

        p_W = torch.einsum('bij,bnj->bnj', self.frames_W_L_[mask][:, :3, :3], p_L) + \
            self.frames_W_L_[mask][:, :3, 3][:, None, ...]
        
        return p_W[..., :2] if d == 2 else p_W


    def world2local(
        self, 
        p_W: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert world points to local points.
        :param p_W: World points, (N, P, 2 / 3).
        :param mask: Mask, (B,), 1 if the b-th heightmap is needed, 0 otherwise.
        :return: Local points, (N, P, 2 / 3).
        """
        assert p_W.ndim == 3 and (p_W.shape[-1] == 2 or p_W.shape[-1] == 3), \
            'p_W should have shape (N, P, 3) or (N, P, 2)'
        N, P, _ = p_W.shape
        assert mask is None or (mask.ndim == 1 and mask.shape[0] == self.B_), \
            'mask should have shape (B,)'
        assert (mask is None and self.B_ == p_W.shape[0]) or \
            (mask is not None and mask.sum() == p_W.shape[0]), \
            '1s in mask should be equal to the number of points batch'
        
        d = p_W.shape[-1]
        if d == 2:  # padding to 3D.
            p_W = torch.cat([p_W, torch.zeros(N, P, 1, device=self.device_)], dim=-1)
        
        if mask is None:
            mask = torch.ones(self.B_, device=self.device_, dtype=torch.bool)

        p_L = torch.einsum('bij,bnj->bnj', self.frames_L_W_[mask][:, :3, :3], p_W) + \
            self.frames_L_W_[mask][:, :3, 3][:, None, ...]
        
        return p_L[..., :2] if d == 2 else p_L

    
    def frames(self) -> torch.Tensor:
        """
        Return the SE(3) frames of the heightmap.
        :return: SE(3) frames, (B, 4, 4).
        """
        return self.frames_W_L_
    

    def bboxes(self) -> torch.Tensor:
        """
        Return the bounding boxes of the heightmap, in local frame.
        :return: Bounding boxes, (B, 2, 2).
        """
        return self.bboxes_
    

    def heights(self) -> torch.Tensor:
        """
        Return the height values of the heightmap.
        :return: Height values, (B, X, Y).
        """
        return self.heights_

    

if __name__ == '__main__':
    B = 2
    frames = torch.eye(4).repeat(B, 1, 1)
    bboxes = torch.tensor([[-5, -5], [5, 5]]).repeat(B, 1, 1)
    heights = torch.rand(B, 1000, 1000)
    heights += 100
    heightmap = Heightmap(frames, bboxes, heights)

    points_W = torch.rand(2, 5, 3)
    mask = torch.tensor([1, 1], device=heightmap.device_, dtype=torch.bool)
    points_L = heightmap.world2local(points_W, mask=mask)

    points_L -= 2
    indices = heightmap.indices_of(points_L, mask=mask, frames='local', return_residue=True)

    height = heightmap.height_at(points_L, mask=mask, frames='local', interpolation='bilinear')
    print(height)