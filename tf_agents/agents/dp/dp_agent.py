
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.typing import types

class DpAgent(object):
  """A Dynamic Programming Agent.
  """
  def __init__(
      self,
      env: py_environment.PyEnvironment,
      dynamics: types.Dict,
      gamma: types.Float = 1.0,
      theta: types.Float = 0.1
  ):
    """Creates a Dynamic Programming Agent.

    Args:
      env (py_environment.PyEnvironment):
        Required. A PyEnvironment instance
          Example:
            from tf_agents.environments import suite_gym
            env = suite_gym.load('FrozenLake-v0', gym_kwargs={'map_name': '8x8'})
      dynamics (types.Dict):
        Required. The dynamics or model of the environment.
          Example:
            dynamics = {state: {action: [] for action in range(num_actions)}
                          for state in range(num_states)}
            dynamics[state][action][transition] = (
                probability, next_state, reward, is_next_state_terminal
              )
      gamma (types.Float):
        A discount factor for future rewards. It must be a value between
        0 and 1, inclusive, default value 1.
      theta (types.Float):
        Convergence threshold, default value 0.1.

    """

    self._env = env

    self._check_observation_spec(env.observation_spec())
    self._check_action_spec(env.action_spec())

    self._dynamics = dynamics

    self._gamma = gamma
    self._theta = theta

    self._values = self._initialize_values()
    self._policy = self._setup_random_policy()

  def _check_observation_spec(
      self,
      observation_spec: types.NestedArraySpec
  ):
    if observation_spec.minimum != 0:
      raise ValueError(
          'Observation specs should have minimum of 0, '
          'but saw: {0}, check the environment'.format(observation_spec)
          )

    self._observation_spec = observation_spec
    self._num_states = observation_spec.maximum - observation_spec.minimum + 1

  def _check_action_spec(
      self,
      action_spec: types.NestedArraySpec
  ):
    if action_spec.minimum != 0:
      raise ValueError(
          'Action specs should have minimum of 0, '
          'but saw: {0}, check the environment'.format(action_spec)
          )

    self._action_spec = action_spec
    self._num_actions = action_spec.maximum - action_spec.minimum + 1

  @property
  def env(self) -> py_environment.PyEnvironment:
    return self._env

  @property
  def dynamics(self) -> types.Dict:
    return self._dynamics

  @property
  def values(self) -> types.Tensor:
    return self._values

  @property
  def policy(self) -> types.Tensor:
    return self._policy

  def reset(self):
    self._values = self._initialize_values()
    self._policy = self._setup_random_policy()

  def _initialize_values(self) -> types.Tensor:
    return tf.zeros(self._num_states)

  def _setup_random_policy(self) -> types.Tensor:
    return tf.ones(shape=(self._num_states, self._num_actions)) / self._num_actions

  def _compute_q_values(
      self,
      state: types.Tensor,
  ) -> types.Tensor:
    q_values = tf.zeros(self._num_actions)
    for action in tf.range(self._num_actions):
      transitions = self._dynamics[state.numpy()][action.numpy()]
      for transition in transitions:
        probability, next_state, reward, is_next_state_terminal = transition
        q_value = q_values[action] + probability * (reward + self._gamma * self._values[next_state])
        q_values = tf.tensor_scatter_nd_update(
            tensor=q_values,
            indices=[[action]],
            updates=[q_value]
        )
    return q_values

  def _bellman_expectation_equation(
      self,
      state: types.Tensor,
  ) -> types.Tensor:
    q_values = self._compute_q_values(state=state)
    return tf.tensordot(self._policy[state], q_values, axes=1)

  def _bellman_optimality_equation(
      self,
      state: types.Tensor,
  ) -> types.Tensor:
    q_values = self._compute_q_values(state=state)
    return tf.reduce_max(q_values)

  def _q_greedify_policy(
      self,
      state: types.Tensor,
      deterministic: types.Bool=False,
  ):
    q_values = self._compute_q_values(state=state)
    self._policy = tf.tensor_scatter_nd_update(
        tensor=self._policy,
        indices=[[state, action] for action in tf.range(self._num_actions)],
        updates=tf.zeros(self._num_actions)
    )

    if deterministic:
      # construct a deterministic policy
      self._policy = tf.tensor_scatter_nd_update(
          tensor=self._policy,
          indices=[[state, tf.argmax(q_values)]],
          updates=[1.]
      )
    else:
      # construct a stochastic policy that puts equal probability on maximizing actions
      best_actions = tf.squeeze(tf.where(q_values == tf.reduce_max(q_values)), axis=1)

      self._policy = tf.tensor_scatter_nd_update(
          tensor=self._policy,
          indices=[[state, action] for action in best_actions],
          updates=tf.ones(shape=(len(best_actions),)) / len(best_actions)
      )

  def policy_evaluation(self):
    delta = float('inf')
    while delta >= self._theta:
      delta = 0
      for state in tf.range(self._num_states):
        old_value = self._values[state]
        value = self._bellman_expectation_equation(state=state)
        self._values = tf.tensor_scatter_nd_update(
            tensor=self._values,
            indices=[[state]],
            updates=[value],
        )
        delta = max(delta, abs(old_value - self._values[state]))

  def policy_improvement(self) -> types.Bool:
    policy_stable = True
    for state in tf.range(self._num_states):
      old_policy = self._policy[state]
      self._q_greedify_policy(state=state)
      if not tf.math.reduce_all(tf.equal(self._policy[state], old_policy)):
        policy_stable = False
    return policy_stable

  def policy_iteration(self):
    policy_stable = False
    while not policy_stable:
      self.policy_evaluation()
      policy_stable = self.policy_improvement()

  def value_iteration(self):
    delta = float('inf')
    while delta >= self._theta:
      delta = 0
      for state in tf.range(self._num_states):
        old_value = self._values[state]
        value = self._bellman_optimality_equation(state=state)
        self._values = tf.tensor_scatter_nd_update(
            tensor=self._values,
            indices=[[state]],
            updates=[value],
        )
        delta = max(delta, abs(old_value - self._values[state]))

    self.policy_improvement()

