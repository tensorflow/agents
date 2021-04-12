
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf

from tf_agents.agents.td import td_agent
from tf_agents.environments import suite_gym

class TestTdAgent:

  def setup_method(self):
    self._grid_size = 4
    self._gym_kwargs = {'map_name': '{}x{}'.format(
        self._grid_size, self._grid_size)}
    self._env = suite_gym.load('FrozenLake-v0', gym_kwargs=self._gym_kwargs)
    self._gamma = 0.9
    self._alpha = 0.01
    self._epsilon = 0.1
    self._episodes = 500
    self._max_step = 500
    self._seed = 1234

class TestSarsaAgent(TestTdAgent):
  @pytest.fixture
  def agent(self):
    agent = td_agent.SarsaAgent(
        env=self._env,
        gamma=self._gamma,
        alpha=self._alpha,
        epsilon=self._epsilon,
        episodes=self._episodes,
        max_step=self._max_step,
        seed=self._seed,
    )
    agent._q_values = tf.constant(
        [[5.60819558e-07, 7.76249124e-07, 1.29604923e-05, 1.61629498e-06],
         [5.71825638e-07, 1.88193849e-06, 5.17496046e-05, 4.85363353e-06],
         [2.52256741e-05, 6.64892068e-05, 2.24431918e-04, 3.20157733e-06],
         [2.18609966e-06, 4.26822635e-06, 4.96758503e-07, 3.85910219e-08]])
    return agent

  @pytest.mark.parametrize(
      "next_state, is_done, next_action, expected_value", [
          (tf.constant(2), False, tf.constant(0), 2.5225674e-05),
          (tf.constant(1), False, tf.constant(3), 4.8536335e-06),
      ]
  )
  def test_get_next_q_value(self, agent, next_state, is_done, next_action, expected_value):
    actual_value = agent._get_next_q_value(next_state, is_done, next_action)
    print(actual_value)
    assert tf.equal(actual_value, expected_value)

class TestQLearningAgent(TestTdAgent):
  @pytest.fixture
  def agent(self):
    agent = td_agent.QLearningAgent(
        env=self._env,
        gamma=self._gamma,
        alpha=self._alpha,
        epsilon=self._epsilon,
        episodes=self._episodes,
        max_step=self._max_step,
        seed=self._seed,
    )
    agent._q_values = tf.constant(
        [[2.4145023e-05, 1.4690059e-04, 1.0814341e-05, 8.7741209e-06],
         [1.1580735e-06, 5.2562267e-05, 4.2573220e-06, 3.9840897e-06],
         [9.3073568e-06, 1.9187988e-04, 2.9009094e-05, 2.0168284e-07],
         [8.5336443e-12, 2.1275366e-05, 2.5598757e-07, 4.2717825e-09]])
    return agent

  @pytest.mark.parametrize(
      "next_state, is_done, next_action, expected_value", [
          (tf.constant(2), False, tf.constant(0), 0.00019187988),
          (tf.constant(1), False, tf.constant(3), 5.2562267e-05),
      ]
  )
  def test_get_next_q_value(self, agent, next_state, is_done, next_action, expected_value):

    actual_value = agent._get_next_q_value(next_state, is_done, next_action)
    assert tf.equal(actual_value, expected_value)

class TestExpectedSarsaAgent(TestTdAgent):
  @pytest.fixture
  def agent(self):
    agent = td_agent.ExpectedSarsaAgent(
        env=self._env,
        gamma=self._gamma,
        alpha=self._alpha,
        epsilon=self._epsilon,
        episodes=self._episodes,
        max_step=self._max_step,
        seed=self._seed,
    )
    agent._q_values = tf.constant(
        [[3.3158765e-06, 4.2852598e-06, 6.3720836e-05, 7.8116063e-06],
         [3.8391777e-06, 2.5867007e-06, 1.7285626e-04, 1.4589940e-05],
         [6.3734886e-05, 4.0375195e-05, 7.5963582e-04, 6.9956652e-05],
         [1.3597673e-05, 9.3515237e-06, 1.9832876e-05, 1.2827848e-04]])
    return agent

  @pytest.mark.parametrize(
      "next_state, is_done, next_action, expected_value", [
          (tf.constant(2), False, tf.constant(0), 0.0007070148),
          (tf.constant(1), False, tf.constant(3), 0.00016041745),
      ]
  )
  def test_get_next_q_value(self, agent, next_state, is_done, next_action, expected_value):
    actual_value = agent._get_next_q_value(next_state, is_done, next_action)
    assert tf.equal(actual_value, expected_value)

if __name__ == '__main__':
  tf.test.main()
