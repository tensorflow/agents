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

"""Tests for tf_agents.bandits.agents.exp3_mixture_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.bandits.agents import exp3_mixture_agent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.policies import mixture_policy
from tf_agents.bandits.policies import policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.utils import test_utils

tfd = tfp.distributions


def _get_initial_and_final_steps(batch_size, context_dim):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
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
      tf.constant(0.5, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(
          observation + 100.0,
          dtype=tf.float32,
          shape=[batch_size, context_dim],
          name='observation'))
  return initial_step, final_step


def _get_action_step(action, num_agents, num_actions, emit_policy_info):
  batch_size = tf.shape(action)[0]
  choices = tf.constant(num_agents - 1, shape=action.shape, dtype=tf.int32)
  predicted_rewards_mean = (
      tf.zeros([batch_size, num_actions]) if emit_policy_info else ())
  return policy_step.PolicyStep(
      action=tf.convert_to_tensor(action),
      info={
          mixture_policy.MIXTURE_AGENT_ID:
              choices,
          mixture_policy.SUBPOLICY_INFO:
              policy_utilities.PolicyInfo(
                  predicted_rewards_mean=predicted_rewards_mean)
      })


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1), single_experience)


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10_numagents2_info',
          'batch_size': 1,
          'context_dim': 10,
          'num_agents': 2,
          'emit_policy_info': True
      }, {
          'testcase_name': '_batch3_contextdim7_numagents17_noinfo',
          'batch_size': 3,
          'context_dim': 7,
          'num_agents': 17,
          'emit_policy_info': False
      }, {
          'testcase_name': '_batch4_contextdim5_numagents10_info',
          'batch_size': 4,
          'context_dim': 5,
          'num_agents': 10,
          'emit_policy_info': True
      })


class Exp3MixtureAgentTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(Exp3MixtureAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()

  @test_cases()
  def testInitializeAgent(self, batch_size, context_dim, num_agents,
                          emit_policy_info):
    num_actions = 7
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agents = [
        lin_ucb_agent.LinearUCBAgent(time_step_spec, action_spec)
        for _ in range(num_agents)
    ]
    mixed_agent = exp3_mixture_agent.Exp3MixtureAgent(agents)
    self.evaluate(mixed_agent.initialize())

  @test_cases()
  def testMixtureUpdate(self, batch_size, context_dim, num_agents,
                        emit_policy_info):
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agents = []
    policy_info = (policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,
                  ) if emit_policy_info else ()
    for _ in range(num_agents):
      agents.append(
          lin_ucb_agent.LinearUCBAgent(
              time_step_spec, action_spec, emit_policy_info=policy_info))
    mixed_agent = exp3_mixture_agent.Exp3MixtureAgent(agents)
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action, num_agents, num_actions,
                                   emit_policy_info)
    if emit_policy_info:
      self.assertEqual(
          self.evaluate(
              action_step.info['subpolicy_info'].predicted_rewards_mean[0, 0]),
          0)
    experience = _get_experience(initial_step, action_step, final_step)
    self.evaluate(mixed_agent.initialize())
    self.evaluate(mixed_agent._variable_collection.reward_aggregates)
    self.evaluate(mixed_agent.train(experience))
    reward_aggregates = self.evaluate(
        mixed_agent._variable_collection.reward_aggregates)
    self.assertAllInSet(reward_aggregates[:num_agents - 1], [0.999])
    agent_prob = 1 / num_agents
    est_rewards = 0.5 / agent_prob
    per_step_update = est_rewards
    last_agent_update = 1 - batch_size * per_step_update
    self.assertAllClose(reward_aggregates[-1], last_agent_update * 0.999)
if __name__ == '__main__':
  tf.test.main()
