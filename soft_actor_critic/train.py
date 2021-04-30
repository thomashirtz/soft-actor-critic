import gym
import torch
import datetime
import numpy as np
from itertools import count
from typing import Optional
from typing import Sequence

from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from utilities import save_to_writer


def train(env_name: str, batch_size: int = 256, memory_size: int = 10e6, start_step: int = 1_000, seed: int = 0,
          learning_rate: float = 3e-4, alpha: float = 0.05, gamma: float = 0.99, tau: float = 0.005,  # todo maybe try 1e-3
          num_steps: int = 1_000_000, hidden_units: Optional[Sequence[int]] = None, load_models: bool = False):

    env = gym.make(env_name)
    observation_shape = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    agent = Agent(observation_shape=observation_shape, num_actions=num_actions,
                  alpha=alpha, learning_rate=learning_rate, gamma=gamma, tau=tau,
                  hidden_units=hidden_units, batch_size=batch_size, memory_size=memory_size)

    writer = SummaryWriter(f'runs/SAC_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)

    updates = 0
    global_step = 0
    score_history = []

    for episode in count():
        score = 0
        done = False
        observation = env.reset()
        episode_step = 0

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
                tensorboard_logs = agent.learn()
                save_to_writer(writer, tensorboard_logs, updates)
                updates += 1

        score_history.append(score)
        average_score = np.mean(score_history[-100:])
        print(f'episode {episode} \tepisode_step {episode_step} \tglobal_step {global_step} \tscore {score:.2f} \ttrailing 100 games avg {average_score}')

        tensorboard_logs = {
            'train/episode_step': episode_step,
            'train/score': score,
            'train/average_score': average_score
        }
        save_to_writer(writer, tensorboard_logs, global_step)  # todo https://github.com/wandb/client/issues/357

        if global_step > num_steps:
            break

        if episode % 20 == 0:
            print('saving models')
            # agent.save_models()


if __name__ == '__main__':
    train('LunarLanderContinuous-v2', load_models=False)
