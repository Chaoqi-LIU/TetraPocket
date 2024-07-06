import torch
from typing import List



def multi2single_gpu_state_dict(state_dict):
    """
    Convert multi-gpu state_dict to single-gpu state_dict.
    :param state_dict: multi-gpu / single-gpu state_dict
    :return: single-gpu state_dict, i.e., no change if is already single-gpu state_dict
    """
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def get_device(allow_gpu: bool = True) -> torch.device:
    """
    Get device.
    :param allow_gpu: whether allow gpu
    :return: device
    """
    if allow_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')
    

def count_all_model_parameters(model: torch.nn.Module) -> int:
    """
    Count all model parameters.
    :param model: model
    :return: number of all parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_model_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable model parameters.
    :param model: model
    :return: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nontrainable_model_parameters(model: torch.nn.Module) -> int:
    """
    Count non-trainable model parameters.
    :param model: model
    :return: number of non-trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> List[float]:
    """
    Get learning rate from optimizer.
    :param optimizer: optimizer
    :return: learning rate
    """
    return [group['lr'] for group in optimizer.param_groups]


class LossMeter:
    """
    Loss meter. Update and query average loss, etc.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count