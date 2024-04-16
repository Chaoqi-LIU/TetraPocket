import numpy as np
import time
from pydrake.geometry.optimization import (
    HPolyhedron,
)
from pydrake.common import (
    RandomGenerator,
)
from typing import (
    Optional,
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