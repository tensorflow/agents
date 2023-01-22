"""Temporal Difference Learning Agents"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional

import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.typing import types

class TdAgent(metaclass=abc.ABCMeta):
  """An abstract Temporal Difference Agent.
  """
  def __init__(
      self,
      env: py_environment.PyEnvironment,
      gamma: types.Float = 1.0,
      alpha: types.Float = 0.01,
      epsilon: types.Float = 0.1,
      episodes: types.Int = 1000,
      max_step: Optional[types.Int] = 1000,
      seed: types.Int = 1234,
  ):
    """Creates a Temporal Difference Agent.

    Args:
      env (py_environment.PyEnvironment):
        Required. A PyEnvironment instance
          Example:
            from tf_agents.environments import suite_gym
            env = suite_gym.load('FrozenLake-v0', gym_kwargs={'map_name': '8x8'})
      gamma (types.Float):
        A discount factor for future rewards. It must be a value between
        0 and 1, inclusive, default value 1.
      alpha (types.Float):
        The step-size parameter for the update step, default value 0.01
      epsilon (types.Float):
        The probability of choosing a random action, default value 0.1
      episodes (types.Int):
        The number of episodes that are generated through agent-environment
        interaction, default value 1000
      max_step (types.Int):
        The max step in each episode, default value 1000
      seed (types.Int):
        The global random seed, default value 1234
    """
    self._env = env

    self._check_observation_spec(env.observation_spec())
    self._check_action_spec(env.action_spec())

    self._q_values = self._initialize_q_values()
    self._policy = self._greedy_q_policy()

    self._gamma = gamma
    self._alpha = alpha
    self._epsilon = epsilon

    self.episodes = episodes
    self._max_step = max_step

    # Create a random number generator for reproducibility.
    self._seed = seed
    self._rand_generator = np.random.RandomState(self._seed)

  @property
  def env(self) -> py_environment.PyEnvironment:
    return self._env

  @property
  def q_values(self) -> types.Tensor:
    return self._q_values

  @property
  def values(self) -> types.Tensor:
    return tf.reduce_max(self._q_values, axis=1)

  @property
  def total_reward(self) -> types.Tensor:
    return self._total_reward

  @property
  @abc.abstractmethod
  def _get_next_q_value(self) -> types.Tensor:
    pass

  def reset(self):
    self._q_values = self._initialize_q_values()
    self._policy = self._greedy_q_policy()

  def run(self):

    self._total_reward = 0.
    self._num_steps = 0
    time_step = self._env.reset()
    state = time_step.observation
    action = self._epsilon_greedy_action_selection(state)

    while True:
      time_step = self._env.step(action=action)
      reward = time_step.reward
      next_state = time_step.observation
      is_next_state_terminal = self._env.done
      self._num_steps += 1
      self._total_reward += reward

      is_done = self._is_done(is_next_state_terminal)
      next_action = None if is_done else self._epsilon_greedy_action_selection(state)
      next_q_value = self._get_next_q_value(
          next_state=next_state,
          is_done=is_done,
          next_action=next_action
      )
      q_value = self._compute_q_value(
          state=state,
          action=action,
          reward=reward,
          next_q_value=next_q_value
      )
      self._q_values = tf.tensor_scatter_nd_update(
          tensor=self._q_values,
          indices=[[state, action]],
          updates=[q_value]
      )
      if is_done:
        break
      state, action = next_state, next_action

  def _check_observation_spec(
      self,
      observation_spec: types.NestedArraySpec
  ):
    if observation_spec.minimum != 0:
      raise ValueError(
          'Observation specs should have minimum of 0, but saw: {0}, check the environment'.format(observation_spec))

    self._observation_spec = observation_spec
    self._num_states = observation_spec.maximum - observation_spec.minimum + 1

  def _check_action_spec(
      self,
      action_spec: types.NestedArraySpec
  ):
    if action_spec.minimum != 0:
      raise ValueError(
          'Action specs should have minimum of 0, but saw: {0}, check the environment'.format(action_spec))

    self._action_spec = action_spec
    self._num_actions = action_spec.maximum - action_spec.minimum + 1

  def _initialize_q_values(self) -> types.Tensor:
    return tf.zeros((self._num_states, self._num_actions))

  @property
  def policy(self) -> types.Tensor:
    return self._policy

  def _greedy_q_policy(self) -> types.Tensor:
    return tf.argmax(self._q_values, axis=1)

  def _is_done(
      self,
      is_next_state_terminal: types.Bool
  ) -> types.Bool:
    is_max_step = False
    if self._max_step and self._num_steps >= self._max_step:
      is_max_step = True
    return is_next_state_terminal or is_max_step

  def _random_action_selection(self) -> types.Tensor:
    return self._rand_generator.randint(self._num_actions)

  def _epsilon_greedy_action_selection(
      self,
      state: types.Tensor,
  ) -> types.Tensor:
    if self._rand_generator.rand() <= self._epsilon:
      return self._random_action_selection()
    best_actions = tf.squeeze(
        tf.where(self._q_values[state] == tf.reduce_max(self._q_values[state])),
        axis=1
    )
    return self._rand_generator.choice(best_actions)

  def _compute_q_value(
      self,
      state: types.Tensor,
      action: types.Tensor,
      reward: types.Float,
      next_q_value: types.Tensor,
  ) -> types.Tensor:
    td_target = reward + self._gamma * next_q_value
    td_error = td_target - self.q_values[state][action]
    q_value = self._q_values[state][action] + self._alpha * td_error
    return q_value

class SarsaAgent(TdAgent):

  def _get_next_q_value(
      self,
      next_state: types.Tensor,
      is_done: types.Bool,
      next_action: types.Tensor = None,
  ) -> types.Tensor:
    next_q_value = 0. if is_done else self._q_values[next_state][next_action]
    return next_q_value

class QLearningAgent(TdAgent):

  def _get_next_q_value(
      self,
      next_state: types.Tensor,
      is_done: types.Bool,
      next_action: types.Tensor = None,
  ) -> types.Tensor:
    next_q_value = 0. if is_done else tf.reduce_max(self._q_values[next_state])
    return next_q_value

class ExpectedSarsaAgent(TdAgent):

  def _get_next_q_value(
      self,
      next_state: types.Tensor,
      is_done: types.Bool,
      next_action: types.Tensor = None,
  ) -> types.Tensor:
    random_prob = self._epsilon / self._num_actions
    policy = tf.ones(self._num_actions) * random_prob
    indices = tf.argmax(self._q_values[next_state])
    policy = tf.tensor_scatter_nd_update(
        tensor=policy,
        indices=[[indices]],
        updates=[random_prob + 1 - self._epsilon]
    )
    next_q_value = 0 if is_done else tf.tensordot(policy, self._q_values[next_state], axes=1)
    return next_q_value

