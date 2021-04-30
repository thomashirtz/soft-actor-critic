from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn

from typing import List
from typing import Union
from typing import Optional


@contextmanager
def eval_mode(net):  # todo add typing
    """Temporarily switch to evaluation mode."""
    is_train = net.training
    try:
        net.eval()
        yield net
    finally:
        if is_train:
            net.train()


def get_device(overwrite: Optional[bool] = None, device: str = 'cuda:0', fallback_device: str = 'cpu') -> torch.device:
    use_cuda = torch.cuda.is_available() if overwrite is None else overwrite
    return torch.device(device if use_cuda else fallback_device)


def update_network_parameters(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_parameters, source_parameters in zip(target.parameters(), source.parameters()):
        target_parameters.data.copy_(target_parameters.data * (1.0 - tau) + source_parameters.data * tau)


def save_model(model: nn.Module, file: Union[str, Path]) -> None:
    torch.save(model.state_dict(), file)


def load_model(model: nn.Module, file: Union[str, Path]) -> None:
    model.load_state_dict(torch.load(file))


def save_to_writer(writer, tag_to_scalar_value: dict, step: int) -> None:  # todo check if it actually needed
    for tag, scalar_value in tag_to_scalar_value.items():
        writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)


def weight_initialization(module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


def get_multilayer_perceptron(unit_list: List[int], keep_last_relu: bool = False) -> nn.Sequential:
    module_list = []
    for in_features, out_features in zip(unit_list, unit_list[1:]):
        module_list.append(nn.Linear(in_features, out_features))
        module_list.append(nn.ReLU())
    if keep_last_relu:
        return nn.Sequential(*module_list)
    else:
        return nn.Sequential(*module_list[:-1])
