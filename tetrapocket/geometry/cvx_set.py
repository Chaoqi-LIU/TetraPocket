import numpy as np
import torch
import time
from pydrake.geometry.optimization import (
    HPolyhedron,
)
from pydrake.common import (
    RandomGenerator,
)
from typing import (
    Optional,
    Union,
    List,
)



def uniform_sample_in_hpolyhedron(
    hpoly: HPolyhedron,
    num_samples: int,
    seed: Optional[int] = None,
    method: Optional[str] = 'mcmc'
) -> np.ndarray:
    """
    Uniform sample in half-space polyhedron.
    :param hpoly: half-space polyhedron, represented by Ax ≤ b
    :param num_samples: number of samples
    :method: either 'mcmc' or 'rejection'
    :return: samples, (num_samples, dim)

    for more details:
    check drake.geometry.optimization.HPolyhedron.UniformSample
        https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1optimization_1_1_h_polyhedron.html
    and this paper:
            https://link.springer.com/article/10.1007/s101070050099
    """
    assert method in ['mcmc', 'rejection'], 'method should be mcmc or rejection'
    assert num_samples > 0, 'num_samples should be positive'

    seed = seed if seed is not None else int(time.time())

    if method == 'mcmc':
        random_gen = RandomGenerator(seed)
        samples = []
        for _ in range(num_samples):
            if len(samples) == 0:
                sample = hpoly.UniformSample(random_gen)
            else:
                sample = hpoly.UniformSample(random_gen, samples[-1])
            samples.append(sample)
        return np.array(samples)

    elif method == 'rejection':
        random_gen = np.random.Generator(np.random.PCG64(seed))
        dim = hpoly.A().shape[1]
        samples = []
        while len(samples) < num_samples:
            sample = random_gen.random(dim)
            if np.all(hpoly.A() @ sample <= hpoly.b()):
                samples.append(sample)
        return np.array(samples)


class Ellipsoid:
    def __init__(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        frames: Optional[torch.Tensor] = None
    ) -> None:
        """
        Ellipsoid in the form of (x - c)^T R^T A R (x - c) ≤ 1.
        :param center: center of the ellipsoid, (B, d)
        :param radii: radii of the ellipsoid, (B, d), 
                      the lengths of the principal semi-axes of the ellipsoid. 
                      The bounding box of the ellipsoid is [-raddi, radii]
        :param frame: rotation matrix of the ellipsoid, (B, d, d)
        """
        assert centers.ndim == radii.ndim == 2, 'center and radii should have shape (B, d)'
        assert frames is None or frames.ndim == 3, 'frame should have shape (B, d, d)'
        assert centers.device == radii.device and \
            (frames is None or centers.device == frames.device), 'device mismatch'
        
        self.device_ = centers.device
        self.B_ = centers.shape[0]
        self.d_ = centers.shape[-1]
        self.centers_ = centers
        self.radii_ = radii
        self.frames_ = torch.eye(self.d_).repeat(self.B_, 1, 1).to(self.device_) \
            if frames is None else frames
        self.RtAR_ = self.frames_.transpose(1, 2) @ torch.diag_embed(1 / radii**2) @ self.frames_


    def points_in_ellipsoid(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if the points are inside the ellipsoid.
        :param points: points to check, (B, N, d)
        :return: mask, (B, N), 1 if inside, 0 otherwise
        """
        assert points.ndim == 3 and points.shape[-1] == self.d_, 'points should have shape (B, N, d)'
        return torch.einsum('bni,bij,bnj->bn', points - self.centers_.unsqueeze(1), 
            self.RtAR_, points - self.centers_.unsqueeze(1)) <= 1


    def frames(self) -> torch.Tensor:
        """
        Rotation matrix of the ellipsoid.
        :return: rotation matrix, (B, d, d)
        """
        return self.frames_
    

    def centers(self) -> torch.Tensor:
        """
        Center of the ellipsoid.
        :return: center, (B, d)
        """
        return self.centers_
        
    
    def volume(self) -> torch.Tensor:
        """
        Volume of the ellipsoid.
        :return: volume, V = 4/3 * pi * prod(radii), (B,)
        """
        return 4/3 * torch.pi * torch.prod(self.radii_, dim=-1)
