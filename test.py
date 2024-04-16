import torch
import numpy as np
# import tetrapocket as tp
from tetrapocket.math import se3
# from tetrapocket import learning
# import tetrapocket as tp

if __name__ == '__main__':
    X0 = torch.rand(2, 4, 4)
    X1 = torch.rand(2, 4, 4)
    print(se3.linear_se3_interpolation(X0, X1, num_segment=torch.tensor([3, 1]), padding_value=np.inf))