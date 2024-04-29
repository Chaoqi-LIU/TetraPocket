import torch
import numpy as np
from typing import (
    Optional,
    Tuple,
    Union,
)



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
        frames: Optional[str] = 'world',
        return_residue: Optional[bool] = False
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
        frames: Optional[str] = 'world',
        interpolation: Optional[str] = 'bilinear'
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