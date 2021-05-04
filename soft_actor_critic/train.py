import gym
import torch
import datetime
import numpy as np
from pathlib import Path
from itertools import count
from typing import Optional
from typing import Sequence

from torch.utils.tensorboard import SummaryWriter

from soft_actor_critic.agent import Agent
from soft_actor_critic.utilities import save_to_writer
from soft_actor_critic.utilities import get_run_name


def train(env_name: str, env_kwargs: Optional[dict] = None, batch_size: int = 256, memory_size: int = 10e6,
          learning_rate: float = 3e-4, alpha: float = 0.05, gamma: float = 0.99, tau: float = 0.005,
          num_steps: int = 1_000_000, hidden_units: Optional[Sequence[int]] = None, load_models: bool = False,
          saving_frequency: int = 20, run_name: Optional[str] = None, start_step: int = 1_000, seed: int = 0,
          updates_per_step: int = 1, checkpoint_directory: str = '../checkpoints/', flush_print: bool = True):

    env_kwargs = env_kwargs or {}
    env = gym.make(env_name, **env_kwargs)
    observation_shape = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    run_name = run_name if run_name is not None else get_run_name('SAC', env_name, lr=learning_rate)
    run_directory = Path(checkpoint_directory) / run_name
    writer = SummaryWriter(run_directory)

    agent = Agent(observation_shape=observation_shape, num_actions=num_actions,
                  alpha=alpha, learning_rate=learning_rate, gamma=gamma, tau=tau,
                  hidden_units=hidden_units, batch_size=batch_size, memory_size=memory_size,
                  checkpoint_directory=run_directory, load_models=load_models)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    updates = 0
    global_step = 0
    score_history = []

    for episode in count():
        score = 0
        done = False
        episode_step = 0
        last_save_step = -1
        observation = env.reset()

        while not done:
            if start_step > global_step:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.choose_action(observation)  # Sample action from policy

            new_observation, reward, done, info = env.step(action)
            episode_step += 1
            global_step += 1

            _done = 0 if episode_step == env._max_episode_steps else int(done)  # noqa
            agent.remember(observation, action, reward, new_observation, _done)

            observation = new_observation
            score += reward

            if agent.memory.memory_counter >= batch_size:
                for update in range(updates_per_step):
                    tensorboard_logs = agent.learn()
                    save_to_writer(writer, tensorboard_logs, updates)
                    updates += 1

        score_history.append(score)
        average_score = np.mean(score_history[-100:])
        print(f'\repisode {episode} \tepisode_step {episode_step} \tglobal_step {global_step} \tscore {score:.3f} '
              f'\ttrailing 100 games avg {average_score:.3f} \t last_save_step {last_save_step}', end="", flush=True)

        tensorboard_logs = {
            'train/episode_step': episode_step,
            'train/score': score,
            'train/average_score': average_score
        }
        save_to_writer(writer, tensorboard_logs, global_step)

        if episode % saving_frequency == 0:
            last_save_step = global_step
            agent.save_models()

        if global_step > num_steps:
            break



