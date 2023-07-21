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

"""Tests for tf_agents.bandits.policies.categorical_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.policies import categorical_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions

TEMP_UPDATE_TEST_INITIAL_INVERSE_TEMP = 1e-3
TEMP_UPDATE_TEST_FINAL_INVERSE_TEMP = 1e3
TEMP_UPDATE_TEST_BATCH_SIZE = 10000


def _get_dummy_observation_step(observation_shape, batch_size):
  obs_spec = tensor_spec.TensorSpec(observation_shape, tf.float32)
  time_step_spec = time_step.time_step_spec(obs_spec)
  return tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(batch_size,))


@test_util.run_all_in_graph_and_eager_modes
class CategoricalPolicyTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(
          observation_shape=[1],
          batch_size=1,
          weights=np.ones(10),
          inverse_temperature=1.),
      dict(
          observation_shape=[2, 1, 3],
          batch_size=32,
          weights=np.arange(17),
          inverse_temperature=10.),
  )
  def testActionShape(self, observation_shape, batch_size, weights,
                      inverse_temperature):
    observation_spec = tensor_spec.TensorSpec(
        shape=observation_shape, dtype=tf.float32, name='observation_spec')
    time_step_spec = time_step.time_step_spec(observation_spec)

    weights = tf.compat.v2.Variable(weights, dtype=tf.float32)
    inverse_temperature = tf.compat.v2.Variable(
        inverse_temperature, dtype=tf.float32)

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=tf.compat.dimension_value(weights.shape[0]) - 1,
        name='action')

    policy = categorical_policy.CategoricalPolicy(weights, time_step_spec,
                                                  action_spec,
                                                  inverse_temperature)
    observation_step = _get_dummy_observation_step(observation_shape,
                                                   batch_size)
    action_time_step = policy.action(observation_step)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(action_time_step.action.shape.as_list(), [batch_size])

  @parameterized.parameters(
      dict(observation_shape=[1], batch_size=1, weights=np.ones(10)),
      dict(observation_shape=[2, 1, 3], batch_size=32, weights=np.arange(17)),
  )
  def testVariableWeightsDefaultTemp(self, observation_shape, batch_size,
                                     weights):
    observation_spec = tensor_spec.TensorSpec(
        shape=observation_shape, dtype=tf.float32, name='observation_spec')
    time_step_spec = time_step.time_step_spec(observation_spec)

    weights = tf.compat.v2.Variable(weights, dtype=tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=tf.compat.dimension_value(weights.shape[0]) - 1,
        name='action')
    policy = categorical_policy.CategoricalPolicy(weights, time_step_spec,
                                                  action_spec)
    observation_step = _get_dummy_observation_step(observation_shape,
                                                   batch_size)
    action_time_step = policy.action(observation_step)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(action_time_step.action.shape.as_list(), [batch_size])

  @parameterized.parameters(
      dict(observation_shape=[1], weights=np.array([0., 1.]), seed=934585),
      dict(observation_shape=[2, 1, 3], weights=np.arange(10), seed=345789),
  )
  def testInverseTempUpdate(self, observation_shape, weights, seed):
    """Test that temperature updates perform as expected as it is increased."""
    observation_spec = tensor_spec.TensorSpec(
        shape=observation_shape, dtype=tf.float32, name='observation_spec')
    time_step_spec = time_step.time_step_spec(observation_spec)

    weight_var = tf.compat.v2.Variable(weights, dtype=tf.float32)
    inverse_temperature_var = tf.compat.v2.Variable(
        TEMP_UPDATE_TEST_INITIAL_INVERSE_TEMP, dtype=tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int64,
        minimum=0,
        maximum=tf.compat.dimension_value(weight_var.shape[0]) - 1,
        name='action')
    policy = categorical_policy.CategoricalPolicy(weight_var, time_step_spec,
                                                  action_spec,
                                                  inverse_temperature_var)
    observation_step = _get_dummy_observation_step(observation_shape,
                                                   TEMP_UPDATE_TEST_BATCH_SIZE)
    tf.compat.v1.set_random_seed(seed)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Set the inverse temperature to a large value.
    self.evaluate(
        tf.compat.v1.assign(inverse_temperature_var,
                            TEMP_UPDATE_TEST_FINAL_INVERSE_TEMP))

    final_action_time_step = self.evaluate(
        policy.action(observation_step, seed=seed))
    self.assertAllEqual(
        final_action_time_step.action,
        np.full([TEMP_UPDATE_TEST_BATCH_SIZE], np.argmax(weights)))

  @parameterized.named_parameters(
      dict(
          testcase_name='_uniform',
          observation_shape=[1],
          batch_size=1,
          weights=np.ones(10, dtype=np.float32),
          inverse_temperature=1.,
          seed=48579),
      dict(
          testcase_name='_low_to_high',
          observation_shape=[3],
          batch_size=32,
          weights=np.linspace(-2, 2, 20, dtype=np.float32),
          inverse_temperature=2.,
          seed=37595),
  )
  def testActionProbabilities(self, observation_shape, batch_size, weights,
                              inverse_temperature, seed):
    observation_spec = tensor_spec.TensorSpec(
        shape=observation_shape, dtype=tf.float32, name='observation_spec')
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=tf.compat.dimension_value(weights.shape[0]) - 1,
        name='action')
    policy = categorical_policy.CategoricalPolicy(weights, time_step_spec,
                                                  action_spec,
                                                  inverse_temperature)
    observation_step = _get_dummy_observation_step(observation_shape,
                                                   batch_size)
    action_time_step = policy.action(observation_step, seed=seed)

    logits = inverse_temperature * weights
    z = tf.reduce_logsumexp(logits)
    expected_logprob = logits - z
    expected_action_prob = tf.exp(
        tf.gather(expected_logprob, action_time_step.action))
    actual_action_prob = tf.exp(
        policy_step.get_log_probability(action_time_step.info))
    expected_action_prob_val, actual_action_prob_val = self.evaluate(
        [expected_action_prob, actual_action_prob])
    self.assertAllClose(expected_action_prob_val, actual_action_prob_val)


if __name__ == '__main__':
  tf.test.main()
