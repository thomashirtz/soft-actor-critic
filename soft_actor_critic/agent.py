import numpy as np

from typing import Optional
from typing import Sequence

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory import ReplayBuffer
from critic import TwinnedQNetworks
from policy import StochasticPolicy

from utilities import eval_mode
from utilities import get_device
from utilities import update_network_parameters


class Agent:
    def __init__(self, observation_shape: int, num_actions: int, batch_size: int = 256, memory_size: int = 10e6,
                 learning_rate: float = 3e-4, alpha: float = 0.2, gamma: float = 0.99, tau: float = 0.005,
                 hidden_units: Optional[Sequence[int]] = None):

        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.num_actions = num_actions
        self.observation_shape = observation_shape

        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = ReplayBuffer(memory_size, self.observation_shape, self.num_actions)

        self.policy = StochasticPolicy(input_dims=self.observation_shape, num_actions=self.num_actions, hidden_units=hidden_units)
        self.critic = TwinnedQNetworks(input_dims=self.observation_shape, num_actions=self.num_actions, hidden_units=hidden_units)
        self.target_critic = TwinnedQNetworks(input_dims=self.observation_shape, num_actions=self.num_actions, hidden_units=hidden_units)

        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.target_critic.eval()

        self.device = get_device()
        for network in [self.policy, self.critic, self.target_critic]:
            network.to(device=self.device)

        update_network_parameters(self.critic, self.target_critic, tau=1)

        self.target_entropy = -torch.Tensor([num_actions]).to(self.device).item()  # todo change to prod
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def choose_action(self, observation) -> np.array:
        with eval_mode(self.policy), torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
            action, _ = self.policy.forward(observation)
        return action.cpu().detach().numpy()[0]  # todo check about the unsqueeze and [0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Critic Improvement
        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(next_state)
            target_next_q_1, target_next_q_2 = self.target_critic.forward(next_state, next_action)
            min_target_next_q = torch.min(target_next_q_1, target_next_q_2)
            next_q = reward + (1 - done) * self.gamma * (min_target_next_q - self.alpha * next_log_pi)

        q_1, q_2 = self.critic.forward(state, action)
        q_1_loss = F.mse_loss(q_1, next_q)
        q_2_loss = F.mse_loss(q_2, next_q)
        q_loss = q_1_loss + q_2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Policy improvement
        pi, log_pi = self.policy.sample(state)
        pi_q_1, pi_q_2 = self.critic(state, pi)
        min_pi_q = torch.min(pi_q_1, pi_q_2)

        policy_loss = ((self.alpha * log_pi) - min_pi_q).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        update_network_parameters(self.critic, self.target_critic, self.tau)

        tensorboard_logs = {
            'loss/critic_1': q_1_loss.item(),
            'loss/critic_2': q_2_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/entropy_loss': alpha_loss.item(),
            'entropy_temperature/alpha': self.alpha.clone().item(),
        }
        return tensorboard_logs

    def save_models(self):
        pass  # todo implement save and load models
