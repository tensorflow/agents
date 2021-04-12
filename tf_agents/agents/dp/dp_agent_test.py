
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import gym
import tensorflow as tf

from tf_agents.agents.dp import dp_agent
from tf_agents.environments import suite_gym

class TestDpAgent:

  def setup_method(self):
    self._grid_size = 4
    self._gym_kwargs = {'map_name': '{}x{}'.format(
                          self._grid_size, self._grid_size)}
    self._env = suite_gym.load('FrozenLake-v0', gym_kwargs=self._gym_kwargs)
    self._gym_env = gym.spec('FrozenLake-v0').make(**self._gym_kwargs)
    self._dynamics = self._gym_env.P
    self._gamma = 0.9
    self._theta = 0.1

  def test_create_DPAgent(self):
    agent = dp_agent.DpAgent(
        env=self._env,
        dynamics=self._dynamics,
        gamma=self._gamma,
        theta=self._theta
    )
    assert agent.values is not None
    assert agent.policy is not None

  @pytest.mark.parametrize(
      "state,expected_values", [
          (tf.constant(6), tf.constant([0., 0., 0., 0.])),
          (tf.constant(14), tf.constant([0., 0.33333333, 0.33333333, 0.33333333])),
      ]
  )
  def test_compute_q_values(self, state, expected_values):
    agent = dp_agent.DpAgent(
        env=self._env,
        dynamics=self._dynamics,
        gamma=self._gamma,
        theta=self._theta
    )
    actual_values = agent._compute_q_values(state=state)
    assert tf.math.reduce_all(tf.equal(actual_values, expected_values))

  @pytest.mark.parametrize(
      "state,expected_value", [
          (tf.constant(6), tf.constant(0.)),
          (tf.constant(14), tf.constant(0.25)),
      ]
  )
  def test_bellman_expectation_equation(self, state, expected_value):
    agent = dp_agent.DpAgent(
        env=self._env,
        dynamics=self._dynamics,
        gamma=self._gamma,
        theta=self._theta
    )
    actual_value = agent._bellman_expectation_equation(state=state)
    assert tf.equal(actual_value, expected_value)


  @pytest.mark.parametrize(
      "state,expected_value", [
          (tf.constant(6), tf.constant(0.)),
          (tf.constant(14), tf.constant(0.33333333)),
      ]
  )
  def test_bellman_optimality_equation(self, state, expected_value):
    agent = dp_agent.DpAgent(
        env=self._env,
        dynamics=self._dynamics,
        gamma=self._gamma,
        theta=self._theta
    )
    actual_value = agent._bellman_optimality_equation(state=state)
    assert tf.equal(actual_value, expected_value)

if __name__ == '__main__':
  tf.test.main()
