import gym
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from typing import Sequence

from soft_actor_critic.agent import Agent


def evaluate(env_name: str, run_name: str, env_kwargs: Optional[dict] = None, num_episodes: int = 100, seed: int = 0,
             hidden_units: Optional[Sequence[int]] = None, checkpoint_directory: str = '../checkpoints/',
             deterministic: bool = False):

    env_kwargs = env_kwargs or {}
    env = gym.make(env_name, **env_kwargs)
    observation_shape = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    run_directory = Path(checkpoint_directory) / run_name

    agent = Agent(observation_shape=observation_shape, num_actions=num_actions, hidden_units=hidden_units,
                  checkpoint_directory=run_directory, load_models=True)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    score_history = []

    for episode in range(num_episodes):
        score = 0
        done = False
        episode_step = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation, deterministically=deterministic)
            new_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)

            score += reward
            episode_step += 1
            observation = new_observation

        score_history.append(score)
        print(f'\rEpisode n°{episode}  Steps: {episode_step} \tScore: {score:.3f} \tMean: {np.mean(score_history):.3f}')

