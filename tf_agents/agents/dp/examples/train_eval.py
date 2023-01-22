
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import numpy as np
import matplotlib.pyplot as plt

import gym

from tf_agents.agents.dp import dp_agent
from tf_agents.environments import suite_gym

def train_eval():

  logging.set_verbosity(logging.INFO)

  # Environment
  grid_size = 4
  gym_kwargs = {'map_name': '{}x{}'.format(grid_size, grid_size)}
  env = suite_gym.load('FrozenLake-v0', gym_kwargs=gym_kwargs)
  gym_env = gym.spec('FrozenLake-v0').make(**gym_kwargs)
  dynamics = gym_env.P

  agent = dp_agent.DpAgent(
      env=env,
      dynamics=dynamics,
      gamma=0.9,
      theta=1e-8,
  )

  dp_algorithms = [
      'policy_evaluation',
      'policy_iteration',
      'value_iteration'
  ]
  for dp_algorithm in dp_algorithms:
    agent.reset()
    getattr(agent, dp_algorithm)()
    logging.info(f'{dp_algorithm} result:')
    logging.info(f'policy: {agent.policy}')
    plot_values(agent.values, title=dp_algorithm, grid_size=grid_size)
  return

def plot_values(V, title, grid_size=4):
  # reshape value function
  V_sq = np.reshape(V, (grid_size, grid_size))

  # plot the state-value function
  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_subplot(111)
  im = ax.imshow(V_sq, cmap='cool')
  for (j,i),label in np.ndenumerate(V_sq):
    ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
  plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
  plt.title(f'State-Value Function for {title}')
  plt.show()

def main(_):

  logging.set_verbosity(logging.INFO)
  train_eval()

  return

if __name__ == '__main__':
  app.run(main)
