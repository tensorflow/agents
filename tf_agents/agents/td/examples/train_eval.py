
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import numpy as np

from tf_agents.agents.td import td_agent
from tf_agents.environments import suite_gym

from tqdm import tqdm
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def train_eval():

  logging.set_verbosity(logging.INFO)

  # Environment
  grid_size = 4
  gym_kwargs = {'map_name': '{}x{}'.format(grid_size, grid_size)}
  env = suite_gym.load('FrozenLake-v0', gym_kwargs=gym_kwargs)

  plot_every = 100
  agent_classes = [
      td_agent.SarsaAgent,
      td_agent.QLearningAgent,
      td_agent.ExpectedSarsaAgent
  ]

  episodes = 1000
  for agent_class in agent_classes:
    agent = agent_class(
        env=env,
        gamma=0.9,
        alpha=0.01,
        epsilon=0.1,
        episodes=episodes,
        max_step=1000,
    )
    tmp_scores = deque(maxlen=plot_every)
    avg_scores = deque(maxlen=episodes)

    for episode in tqdm(range(1, episodes+1)):
      agent.epsilon = 1.0 / episode
      agent.run()
      tmp_scores.append(agent.total_reward)
      if (episode % plot_every == 0):
        avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0, episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    logging.info(f'Best Average Reward over {plot_every} Episodes: {np.max(avg_scores)}')

    logging.info(f'{agent.__class__.__name__}')
    logging.info('\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):')
    # print(f'{agent.policy.reshape((grid_size, grid_size))}')
    plot_values(agent.values, (grid_size, grid_size))

  return


def plot_values(values, resize_shape):
  # reshape the state-value function
  values= np.reshape(values, resize_shape)
  # plot the state-value function
  fig = plt.figure(figsize=(15,5))
  ax = fig.add_subplot(111)
  im = ax.imshow(values, cmap='cool')
  for (j,i),label in np.ndenumerate(values):
    ax.text(i, j, np.round(label,3), ha='center', va='center', fontsize=14)
  plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
  plt.title('State-Value Function')
  plt.show()


def main(_):

  logging.set_verbosity(logging.INFO)
  train_eval()

  return

if __name__ == '__main__':
  app.run(main)
