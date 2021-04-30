import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from typing import Optional
from typing import Sequence

from utilities import weight_initialization
from utilities import get_multilayer_perceptron


EPSILON = 10e-6
LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20


class StochasticPolicy(nn.Module):
    def __init__(self, input_dims: int, num_actions: int, hidden_units: Optional[Sequence[int]] = None,
                 action_space=None):
        super(StochasticPolicy, self).__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        units = [input_dims] + list(hidden_units)
        self.multilayer_perceptron = get_multilayer_perceptron(units, keep_last_relu=True)

        self.mean_linear = nn.Linear(units[-1], num_actions)
        self.log_std_linear = nn.Linear(units[-1], num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.apply(weight_initialization)

    def forward(self, x):
        x = self.multilayer_perceptron(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std_clamped = torch.clamp(log_std, min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)
        std = torch.exp(log_std_clamped)
        return mean, std

    def sample(self, state):  # todo add mean and evaluate
        mean, std = self.forward(state)
        distribution = Normal(mean, std)
        u = distribution.rsample()
        action = torch.tanh(u) * self.action_scale + self.action_bias

        log_probability = distribution.log_prob(u) - torch.log(self.action_scale * (1 - action.pow(2)) + EPSILON)
        log_probability = log_probability.sum(1, keepdim=True)

        return action, log_probability
