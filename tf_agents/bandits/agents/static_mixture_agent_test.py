# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tf_agents.bandits.agents.static_mixture_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.agents import static_mixture_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.policies import mixture_policy
from tf_agents.bandits.policies import policy_utilities
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10_numagents2',
          'batch_size': 1,
          'context_dim': 10,
          'num_agents': 2,
      }, {
          'testcase_name': '_batch4_contextdim5_numagents10',
          'batch_size': 4,
          'context_dim': 5,
          'num_agents': 10,
      })


def _get_initial_and_final_steps(batch_size, context_dim):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  initial_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(
          observation,
          dtype=tf.float32,
          shape=[batch_size, context_dim],
          name='observation'))
  final_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(
          observation + 100.0,
          dtype=tf.float32,
          shape=[batch_size, context_dim],
          name='observation'))
  return initial_step, final_step


def _get_action_step(action, num_agents, num_actions):
  batch_size = tf.shape(action)[0]
  choices = tf.random.uniform(
      shape=tf.shape(action), minval=0, maxval=num_agents - 1, dtype=tf.int32)
  return policy_step.PolicyStep(
      action=tf.convert_to_tensor(action),
      info={
          mixture_policy.MIXTURE_AGENT_ID:
              choices,
          mixture_policy.SUBPOLICY_INFO:
              policy_utilities.PolicyInfo(
                  predicted_rewards_mean=tf.zeros([batch_size, num_actions]))
      })


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1), single_experience)


@test_util.run_all_in_graph_and_eager_modes
class StaticMixtureAgentTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(StaticMixtureAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()

  @test_cases()
  def testInitializeAgent(self, batch_size, context_dim, num_agents):
    num_actions = 7
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agents = [
        lin_ucb_agent.LinearUCBAgent(time_step_spec, action_spec)
        for _ in range(num_agents)
    ]
    mixture_agent = static_mixture_agent.StaticMixtureAgent([1] * num_agents,
                                                            agents)
    self.evaluate(mixture_agent.initialize())

  @test_cases()
  def testAgentUpdate(self, batch_size, context_dim, num_agents):
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agents = []
    for _ in range(num_agents):
      agents.append(
          lin_ucb_agent.LinearUCBAgent(
              time_step_spec,
              action_spec,
              emit_policy_info=(
                  policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,)))
    mixture_agent = static_mixture_agent.StaticMixtureAgent([1] * num_agents,
                                                            agents)
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action, num_agents, num_actions)
    experience = _get_experience(initial_step, action_step, final_step)
    for agent in agents:
      self.evaluate(agent.initialize())
    self.evaluate(mixture_agent.initialize())
    loss_info = mixture_agent.train(experience)
    self.evaluate(loss_info)

  def testAgentWithDifferentSubagentsUpdate(self):
    num_actions = 3
    context_dim = 2
    batch_size = 7
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent1 = lin_ucb_agent.LinearUCBAgent(
        time_step_spec,
        action_spec,
        emit_policy_info=(policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,))
    reward_net = q_network.QNetwork(
        input_tensor_spec=observation_spec,
        action_spec=action_spec,
        fc_layer_params=(4, 3, 2))
    agent2 = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec,
        action_spec,
        reward_network=reward_net,
        emit_policy_info=(policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,),
        optimizer=tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.1),
        epsilon=0.1)
    agents = [agent1, agent2]
    mixture_agent = static_mixture_agent.StaticMixtureAgent([1, 1], agents)
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action, 2, num_actions)
    experience = _get_experience(initial_step, action_step, final_step)
    for agent in agents:
      self.evaluate(agent.initialize())
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.evaluate(mixture_agent.initialize())
    loss_info = mixture_agent.train(experience)
    self.evaluate(loss_info)

  @test_cases()
  def testDynamicPartitionOfNestedTensors(self, batch_size, context_dim,
                                          num_agents):
    tensor1 = tf.reshape(
        tf.range(batch_size * context_dim), shape=[batch_size, context_dim])
    tensor2 = tf.reshape(
        tf.range(batch_size * num_agents), shape=[batch_size, num_agents])
    nested_structure = [{'a': tensor1}, tensor2]
    partition_array = [0, 1] * (batch_size // 2) + [0] * (batch_size % 2)
    partition = tf.constant(partition_array, dtype=tf.int32)
    partitioned = static_mixture_agent._dynamic_partition_of_nested_tensors(
        nested_structure, partition, num_agents)
    evaluated = self.evaluate(partitioned)
    self.assertLen(partitioned, num_agents)
    for k in range(num_agents):
      tf.nest.assert_same_structure(evaluated[k], nested_structure)
    self.assertAllEqual(evaluated[0][0]['a'].shape,
                        [(batch_size + 1) // 2, context_dim])
    self.assertAllEqual(evaluated[1][0]['a'].shape,
                        [batch_size // 2, context_dim])
    self.assertAllEqual(evaluated[0][1].shape,
                        [(batch_size + 1) // 2, num_agents])
    self.assertAllEqual(evaluated[1][1].shape, [batch_size // 2, num_agents])


if __name__ == '__main__':
  tf.test.main()
