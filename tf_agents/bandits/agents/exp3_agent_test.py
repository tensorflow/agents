# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tf_agents.bandits.agents.exp3_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import exp3_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step

tfd = tfp.distributions


def _get_initial_and_final_steps(observation_shape, reward):
  batch_size = reward.shape[0]
  time_step_spec = time_step.time_step_spec(
      tensor_spec.TensorSpec(observation_shape, tf.float32))
  initial_step = tensor_spec.sample_spec_nest(
      time_step_spec, outer_dims=(batch_size,))
  final_step = initial_step._replace(reward=tf.convert_to_tensor(reward))
  return initial_step, final_step


def _get_action_step(action, log_prob):
  step = policy_step.PolicyStep(action=tf.convert_to_tensor(action))
  return step._replace(
      info=policy_step.set_log_probability(step.info,
                                           tf.convert_to_tensor(log_prob)))


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1),
      single_experience)


class Exp3AgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(Exp3AgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()

  @parameterized.named_parameters(
      dict(testcase_name='_trivial',
           observation_shape=[],
           num_actions=1,
           learning_rate=1.),
      dict(testcase_name='_2_2_obs',
           observation_shape=[2, 2],
           num_actions=5,
           learning_rate=.3),
  )
  def testInitializeAgent(self,
                          observation_shape,
                          num_actions,
                          learning_rate):
    time_step_spec = time_step.time_step_spec(
        tensor_spec.TensorSpec(observation_shape, tf.float32))
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = exp3_agent.Exp3Agent(time_step_spec=time_step_spec,
                                 action_spec=action_spec,
                                 learning_rate=learning_rate)
    self.evaluate(agent.initialize())

  @parameterized.parameters(
      dict(values=np.array([0.]),
           partitions=np.array([0]),
           num_partitions=1,
           expected_output=np.array([0.])),
      dict(values=np.array([0, 1, 2, 3, 4, 5]),
           partitions=np.array([0, 1, 1, 1, 0, 0]),
           num_partitions=2,
           expected_output=np.array([9, 6])),
      )
  def testSelectiveSum(
      self, values, partitions, num_partitions, expected_output):
    actual_output = exp3_agent.selective_sum(values, partitions, num_partitions)
    self.assertAllCloseAccordingToType(expected_output, actual_output)

  @parameterized.parameters(
      dict(shape=[],
           seed=1234),
      dict(shape=[100],
           seed=2345),
      dict(shape=[3, 4, 3, 6, 2],
           seed=3456),
      )
  def testExp3UpdateValueShape(self, shape, seed):
    tf.compat.v1.set_random_seed(seed)
    reward = tfd.Uniform(0., 1.).sample(shape)
    log_prob = tfd.Normal(0., 1.).sample(shape)
    update_value = exp3_agent.exp3_update_value(reward, log_prob)
    self.assertAllEqual(shape, update_value.shape)

  @parameterized.named_parameters(
      dict(testcase_name='_trivial',
           observation_shape=[],
           num_actions=1,
           action=np.array([0], dtype=np.int32),
           log_prob=np.array([0.], dtype=np.float32),
           reward=np.array([0.], dtype=np.float32),
           learning_rate=1.),
      dict(testcase_name='_8_rewards',
           observation_shape=[2, 2],
           num_actions=5,
           action=np.array([0, 1, 1, 2, 4, 3, 4, 3], dtype=np.int32),
           log_prob=np.log([.1, .2, .2, .4, .6, .2, .4, .2], dtype=np.float32),
           reward=np.array([0., .4, .3, .4, .9, .4, .5, .3], dtype=np.float32),
           learning_rate=.3),
      )
  def testExp3Update(self,
                     observation_shape,
                     num_actions,
                     action,
                     log_prob,
                     reward,
                     learning_rate):
    """Check EXP3 updates for specified actions and rewards."""

    # Compute expected update for each action.
    expected_update_value = exp3_agent.exp3_update_value(reward, log_prob)
    expected_update = np.zeros(num_actions)
    for a, u in zip(action, self.evaluate(expected_update_value)):
      expected_update[a] += u

    # Construct a `Trajectory` for the given action, log prob and reward.
    time_step_spec = time_step.time_step_spec(
        tensor_spec.TensorSpec(observation_shape, tf.float32))
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    initial_step, final_step = _get_initial_and_final_steps(
        observation_shape, reward)
    action_step = _get_action_step(action, log_prob)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update. Record initial and final
    # weights.
    agent = exp3_agent.Exp3Agent(time_step_spec=time_step_spec,
                                 action_spec=action_spec,
                                 learning_rate=learning_rate)
    self.evaluate(agent.initialize())
    initial_weights = self.evaluate(agent.weights)
    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_weights = self.evaluate(agent.weights)
    update = final_weights - initial_weights

    # Check that the actual update matches expectations.
    self.assertAllClose(expected_update, update)


if __name__ == '__main__':
  tf.test.main()
