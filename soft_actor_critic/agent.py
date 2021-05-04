import numpy as np
from pathlib import Path
from pathlib import PurePath

from typing import Optional
from typing import Sequence
from typing import Union

import torch
import torch.optim as optim
import torch.nn.functional as F

from soft_actor_critic.memory import ReplayBuffer
from soft_actor_critic.critic import TwinnedQNetworks
from soft_actor_critic.policy import StochasticPolicy

from soft_actor_critic.utilities import eval_mode
from soft_actor_critic.utilities import get_device
from soft_actor_critic.utilities import save_model
from soft_actor_critic.utilities import load_model
from soft_actor_critic.utilities import update_network_parameters


class Agent:
    def __init__(self, observation_shape: int, num_actions: int, batch_size: int = 256, memory_size: int = 10e6,
                 learning_rate: float = 3e-4, alpha: float = 1, gamma: float = 0.99, tau: float = 0.005,
                 hidden_units: Optional[Sequence[int]] = None, load_models: bool = False,
                 checkpoint_directory: Optional[Union[Path, str]] = None):

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

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.target_critic.eval()

        self.device = get_device()
        for network in [self.policy, self.critic, self.target_critic]:
            network.to(device=self.device)

        update_network_parameters(self.critic, self.target_critic, tau=1)

        self.target_entropy = -torch.Tensor([num_actions]).to(self.device).item()  # todo change to prod
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        self.policy_checkpoint_path = None
        self.critic_checkpoint_path = None

        if checkpoint_directory:
            if not isinstance(checkpoint_directory, PurePath):
                checkpoint_directory = Path(Path)
            checkpoint_directory.mkdir(parents=True, exist_ok=True)

            self.policy_checkpoint_path = checkpoint_directory / 'actor.pt'
            self.critic_checkpoint_path = checkpoint_directory / 'critic.pt'
            if load_models:
                load_model(self.policy, self.policy_checkpoint_path)
                load_model(self.critic, self.critic_checkpoint_path)

    def choose_action(self, observation, evaluate: bool = False) -> np.array:
        with eval_mode(self.policy), torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
            action, _ = self.policy.evaluate(observation, deterministic=evaluate, with_log_probability=False)
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):  # todo split into different methods, optimize alpha, optimize critic, optimize policy etc
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Critic Improvement
        with torch.no_grad():
            next_action, next_log_pi = self.policy.evaluate(next_state)
            next_q_target_1, next_q_target_2 = self.target_critic.forward(next_state, next_action)
            min_next_q_target = torch.min(next_q_target_1, next_q_target_2)
            next_q = reward + (1 - done) * self.gamma * (min_next_q_target - self.alpha * next_log_pi)

        q_1, q_2 = self.critic.forward(state, action)
        q_1_loss = F.mse_loss(q_1, next_q)
        q_2_loss = F.mse_loss(q_2, next_q)
        q_loss = (q_1_loss + q_2_loss) / 2

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        with eval_mode(self.policy):
            # Policy improvement
            pi, log_pi = self.policy.evaluate(state)  # todo maybe add logpi to logging
            pi_q_1, pi_q_2 = self.critic(state, pi)  # todo double check not mean but samples
            min_pi_q = torch.min(pi_q_1, pi_q_2)  # todo alpha is computed before the policy loss

            policy_loss = ((self.alpha * log_pi) - min_pi_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

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
        try:
            save_model(self.policy, self.policy_checkpoint_path)
            save_model(self.critic, self.critic_checkpoint_path)
        except Exception:
            print("Unable to save the models")

    def load_models(self):
        try:
            save_model(self.policy, self.policy_checkpoint_path)
            save_model(self.critic, self.critic_checkpoint_path)
        except Exception:
            print("Unable to load the models")
