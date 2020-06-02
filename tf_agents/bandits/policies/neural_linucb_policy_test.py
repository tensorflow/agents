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

"""Tests for tf_agents.bandits.policies.neural_linucb_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.networks import global_and_arm_feature_network as arm_network
from tf_agents.bandits.policies import neural_linucb_policy
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


_POLICY_VARIABLES_OFFSET = 10.0


class DummyNet(network.Network):

  def __init__(self, observation_spec, obs_dim=2, encoding_dim=10):
    super(DummyNet, self).__init__(observation_spec, (), 'DummyNet')

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            encoding_dim,
            kernel_initializer=tf.compat.v1.initializers.constant(
                np.ones([obs_dim, encoding_dim])),
            bias_initializer=tf.compat.v1.initializers.constant(
                np.zeros([encoding_dim])))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


def get_reward_layer(num_actions=5, encoding_dim=10):
  return tf.keras.layers.Dense(
      num_actions,
      activation=None,
      kernel_initializer=tf.compat.v1.initializers.constant(
          np.ones([encoding_dim, num_actions])),
      bias_initializer=tf.compat.v1.initializers.constant(
          np.array(range(num_actions))))


def get_per_arm_reward_layer(encoding_dim=10):
  return tf.keras.layers.Dense(
      units=1,
      activation=None,
      use_bias=False,
      kernel_initializer=tf.compat.v1.initializers.constant(
          list(range(encoding_dim))))


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_numtrainsteps0',
          'batch_size': 1,
          'actions_from_reward_layer': False,
      }, {
          'testcase_name': '_batch4_numtrainsteps10',
          'batch_size': 4,
          'actions_from_reward_layer': True,
      })


@test_util.run_all_in_graph_and_eager_modes
class NeuralLinUCBPolicyTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(NeuralLinUCBPolicyTest, self).setUp()
    self._obs_dim = 2
    self._obs_spec = tensor_spec.TensorSpec([self._obs_dim], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._num_actions = 5
    self._alpha = 1.0
    self._action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=self._num_actions - 1,
        name='action')
    self._encoding_dim = 10

  @property
  def _a(self):
    a_for_one_arm = 1.0 + 4.0 * tf.eye(self._encoding_dim, dtype=tf.float32)
    return [a_for_one_arm] * self._num_actions

  @property
  def _a_numpy(self):
    a_for_one_arm = 1.0 + 4.0 * np.eye(self._encoding_dim, dtype=np.float32)
    return [a_for_one_arm] * self._num_actions

  @property
  def _b(self):
    return [tf.constant(r * np.ones(self._encoding_dim), dtype=tf.float32)
            for r in range(self._num_actions)]

  @property
  def _b_numpy(self):
    return [np.array([r * np.ones(self._encoding_dim)], dtype=np.float32)
            for r in range(self._num_actions)]

  @property
  def _num_samples_per_arm(self):
    a_for_one_arm = tf.constant([1], dtype=tf.float32)
    return [a_for_one_arm] * self._num_actions

  def _time_step_batch(self, batch_size):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, dtype=tf.int32, shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        tf.constant(np.array(range(batch_size * self._obs_dim)),
                    dtype=tf.float32, shape=[batch_size, self._obs_dim],
                    name='observation'))

  def _time_step_batch_with_mask(self, batch_size):
    observation = tf.constant(
        np.array(range(batch_size * self._obs_dim)),
        dtype=tf.float32,
        shape=[batch_size, self._obs_dim])
    mask = tf.eye(batch_size, num_columns=self._num_actions, dtype=tf.int32)
    observation_with_mask = (observation, mask)
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST,
            dtype=tf.int32,
            shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        observation_with_mask)

  def _per_arm_time_step_batch(self, batch_size, global_obs_dim, arm_obs_dim):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST,
            dtype=tf.int32,
            shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        {
            bandit_spec_utils.GLOBAL_FEATURE_KEY:
                tf.constant(
                    np.array(range(batch_size * global_obs_dim)),
                    dtype=tf.float32,
                    shape=[batch_size, global_obs_dim],
                    name='observation'),
            bandit_spec_utils.PER_ARM_FEATURE_KEY:
                tf.constant(
                    np.array(
                        range(batch_size * self._num_actions * arm_obs_dim)),
                    dtype=tf.float32,
                    shape=[batch_size, self._num_actions, arm_obs_dim],
                    name='observation')
        })

  def _get_predicted_rewards_from_linucb(self, observation_numpy, batch_size):
    """Runs one step of LinUCB using numpy arrays."""
    observation_numpy.reshape([batch_size, self._encoding_dim])

    predicted_rewards = []
    for k in range(self._num_actions):
      a_inv = np.linalg.inv(self._a_numpy[k] + np.eye(self._encoding_dim))
      theta = np.matmul(
          a_inv, self._b_numpy[k].reshape([self._encoding_dim, 1]))
      est_mean_reward = np.matmul(observation_numpy, theta)
      predicted_rewards.append(est_mean_reward)
    predicted_rewards_array = np.stack(
        predicted_rewards, axis=-1).reshape(batch_size, self._num_actions)
    return predicted_rewards_array

  def _get_predicted_rewards_from_per_arm_linucb(self, observation_numpy,
                                                 batch_size):
    """Runs one step of LinUCB using numpy arrays."""
    observation_numpy.reshape(
        [batch_size, self._num_actions, self._encoding_dim])

    predicted_rewards = []
    for k in range(self._num_actions):
      a_inv = np.linalg.inv(self._a_numpy[0] + np.eye(self._encoding_dim))
      theta = np.matmul(
          a_inv, self._b_numpy[0].reshape([self._encoding_dim, 1]))
      est_mean_reward = np.matmul(observation_numpy[:, k, :], theta)
      predicted_rewards.append(est_mean_reward)
    predicted_rewards_array = np.stack(
        predicted_rewards, axis=-1).reshape((batch_size, self._num_actions))
    return predicted_rewards_array

  @test_cases()
  def testBuild(self, batch_size, actions_from_reward_layer):
    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        DummyNet(self._obs_spec),
        self._encoding_dim,
        get_reward_layer(),
        actions_from_reward_layer=actions_from_reward_layer,
        cov_matrix=self._a,
        data_vector=self._b,
        num_samples=self._num_samples_per_arm,
        epsilon_greedy=0.0,
        time_step_spec=self._time_step_spec)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)

  @test_cases()
  def testObservationShapeMismatch(self, batch_size, actions_from_reward_layer):
    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        DummyNet(self._obs_spec),
        self._encoding_dim,
        get_reward_layer(),
        actions_from_reward_layer=actions_from_reward_layer,
        cov_matrix=self._a,
        data_vector=self._b,
        num_samples=self._num_samples_per_arm,
        epsilon_greedy=0.0,
        time_step_spec=self._time_step_spec)

    current_time_step = ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, dtype=tf.int32, shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        tf.constant(np.array(range(batch_size * (self._obs_dim + 1))),
                    dtype=tf.float32, shape=[batch_size, self._obs_dim + 1],
                    name='observation'))
    if tf.executing_eagerly():
      error_type = tf.errors.InvalidArgumentError
      regexp = r'Matrix size-incompatible: In\[0\]: \[%d,3\]' % batch_size
    else:
      error_type = ValueError
      regexp = r'with shape \[%d, 3\]' % batch_size
    with self.assertRaisesRegex(error_type, regexp):
      policy.action(current_time_step)

  @test_cases()
  def testActionBatch(self, batch_size, actions_from_reward_layer):

    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        DummyNet(self._obs_spec),
        self._encoding_dim,
        get_reward_layer(),
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=self._a,
        data_vector=self._b,
        num_samples=self._num_samples_per_arm,
        epsilon_greedy=0.0,
        time_step_spec=self._time_step_spec)

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action_fn = common.function_in_tf1()(policy.action)
    action_step = action_fn(self._time_step_batch(batch_size=batch_size))
    actions_ = self.evaluate(action_step.action)
    self.assertAllGreaterEqual(actions_, self._action_spec.minimum)
    self.assertAllLessEqual(actions_, self._action_spec.maximum)

  @test_cases()
  def testActionBatchWithMask(self, batch_size, actions_from_reward_layer):
    obs_spec = (tensor_spec.TensorSpec([self._obs_dim], tf.float32),
                tensor_spec.TensorSpec([self._num_actions], tf.int32))
    time_step_spec = ts.time_step_spec(obs_spec)
    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        DummyNet(obs_spec[0]),
        self._encoding_dim,
        get_reward_layer(),
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=self._a,
        data_vector=self._b,
        num_samples=self._num_samples_per_arm,
        epsilon_greedy=0.5,
        time_step_spec=time_step_spec,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]))

    action_fn = common.function_in_tf1()(policy.action)
    action_step = action_fn(
        self._time_step_batch_with_mask(batch_size=batch_size))
    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, range(batch_size))

  @test_cases()
  def testActionBatchWithVariablesAndPolicyUpdate(
      self, batch_size, actions_from_reward_layer):

    a_list = []
    a_new_list = []
    b_list = []
    b_new_list = []
    num_samples_list = []
    num_samples_new_list = []
    for k in range(1, self._num_actions + 1):
      a_initial_value = k + 1 + 2 * k * tf.eye(
          self._encoding_dim, dtype=tf.float32)
      a_for_one_arm = tf.compat.v2.Variable(a_initial_value)
      a_list.append(a_for_one_arm)
      b_initial_value = tf.constant(
          k * np.ones(self._encoding_dim), dtype=tf.float32)
      b_for_one_arm = tf.compat.v2.Variable(b_initial_value)
      b_list.append(b_for_one_arm)
      num_samples_initial_value = tf.constant([1], dtype=tf.float32)
      num_samples_for_one_arm = tf.compat.v2.Variable(num_samples_initial_value)
      num_samples_list.append(num_samples_for_one_arm)

      # Variables for the new policy (they differ by an offset).
      a_new_for_one_arm = tf.compat.v2.Variable(
          a_initial_value + _POLICY_VARIABLES_OFFSET)
      a_new_list.append(a_new_for_one_arm)
      b_new_for_one_arm = tf.compat.v2.Variable(
          b_initial_value + _POLICY_VARIABLES_OFFSET)
      b_new_list.append(b_new_for_one_arm)
      num_samples_for_one_arm_new = tf.compat.v2.Variable(
          num_samples_initial_value + _POLICY_VARIABLES_OFFSET)
      num_samples_new_list.append(num_samples_for_one_arm_new)

    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        encoding_network=DummyNet(self._obs_spec),
        encoding_dim=self._encoding_dim,
        reward_layer=get_reward_layer(),
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=a_list,
        data_vector=b_list,
        num_samples=num_samples_list,
        epsilon_greedy=0.0,
        time_step_spec=self._time_step_spec)

    new_policy = neural_linucb_policy.NeuralLinUCBPolicy(
        encoding_network=DummyNet(self._obs_spec),
        encoding_dim=self._encoding_dim,
        reward_layer=get_reward_layer(),
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=a_new_list,
        data_vector=b_new_list,
        num_samples=num_samples_new_list,
        epsilon_greedy=0.0,
        time_step_spec=self._time_step_spec)

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    new_action_step = new_policy.action(
        self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(new_policy.update(policy))

    action_fn = common.function_in_tf1()(policy.action)
    action_step = action_fn(self._time_step_batch(batch_size=batch_size))
    new_action_fn = common.function_in_tf1()(new_policy.action)
    new_action_step = new_action_fn(
        self._time_step_batch(batch_size=batch_size))

    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  @test_cases()
  def testPredictedRewards(
      self, batch_size, actions_from_reward_layer):
    dummy_net = DummyNet(self._obs_spec)
    reward_layer = get_reward_layer()

    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        dummy_net,
        self._encoding_dim,
        reward_layer,
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=self._a,
        data_vector=self._b,
        num_samples=self._num_samples_per_arm,
        epsilon_greedy=0.0,
        time_step_spec=self._time_step_spec,
        emit_policy_info=('predicted_rewards_mean',))

    current_time_step = self._time_step_batch(batch_size=batch_size)
    action_step = policy.action(current_time_step)
    self.assertEqual(action_step.action.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action_fn = common.function_in_tf1()(policy.action)
    action_step = action_fn(current_time_step)

    input_observation = current_time_step.observation
    encoded_observation, _ = dummy_net(input_observation)
    predicted_rewards_from_reward_layer = reward_layer(encoded_observation)
    if actions_from_reward_layer:
      predicted_rewards_expected = self.evaluate(
          predicted_rewards_from_reward_layer)
    else:
      observation_numpy = self.evaluate(encoded_observation)
      predicted_rewards_expected = self._get_predicted_rewards_from_linucb(
          observation_numpy, batch_size)

    p_info = self.evaluate(action_step.info)
    self.assertEqual(p_info.predicted_rewards_mean.dtype, np.float32)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected)

  @test_cases()
  def testPerArmObservation(self, batch_size, actions_from_reward_layer):
    global_obs_dim = 7
    arm_obs_dim = 3
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        global_obs_dim, arm_obs_dim, self._num_actions)
    time_step_spec = ts.time_step_spec(obs_spec)
    dummy_net = arm_network.create_feed_forward_common_tower_network(
        obs_spec,
        global_layers=(3, 4, 5),
        arm_layers=(3, 2),
        common_layers=(4, 3),
        output_dim=self._encoding_dim)
    reward_layer = get_per_arm_reward_layer(encoding_dim=self._encoding_dim)

    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        dummy_net,
        self._encoding_dim,
        reward_layer,
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=self._a[0:1],
        data_vector=self._b[0:1],
        num_samples=self._num_samples_per_arm[0:1],
        epsilon_greedy=0.0,
        time_step_spec=time_step_spec,
        accepts_per_arm_features=True,
        emit_policy_info=('predicted_rewards_mean',))

    current_time_step = self._per_arm_time_step_batch(
        batch_size=batch_size,
        global_obs_dim=global_obs_dim,
        arm_obs_dim=arm_obs_dim)
    action_step = policy.action(current_time_step)
    self.assertEqual(action_step.action.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action_fn = common.function_in_tf1()(policy.action)
    action_step = action_fn(current_time_step)

    input_observation = current_time_step.observation
    encoded_observation, _ = dummy_net(input_observation)

    if actions_from_reward_layer:
      predicted_rewards_from_reward_layer = reward_layer(encoded_observation)
      predicted_rewards_expected = self.evaluate(
          predicted_rewards_from_reward_layer).reshape((-1, self._num_actions))
    else:
      observation_numpy = self.evaluate(encoded_observation)
      predicted_rewards_expected = (
          self._get_predicted_rewards_from_per_arm_linucb(
              observation_numpy, batch_size))

    p_info = self.evaluate(action_step.info)
    self.assertEqual(p_info.predicted_rewards_mean.dtype, np.float32)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected)

  @test_cases()
  def testSparseObs(self, batch_size, actions_from_reward_layer):
    obs_spec = {
        'global': {'sport': tensor_spec.TensorSpec((), tf.string)},
        'per_arm': {
            'name': tensor_spec.TensorSpec((3,), tf.string),
            'fruit': tensor_spec.TensorSpec((3,), tf.string)
        }
    }
    columns_a = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'name', ['bob', 'george', 'wanda']))
    columns_b = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'fruit', ['banana', 'kiwi', 'pear']))
    columns_c = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'sport', ['bridge', 'chess', 'snooker']))

    dummy_net = arm_network.create_feed_forward_common_tower_network(
        obs_spec,
        global_layers=(3, 4, 5),
        arm_layers=(3, 2),
        common_layers=(4, 3),
        output_dim=self._encoding_dim,
        global_preprocessing_combiner=(tf.compat.v2.keras.layers.DenseFeatures(
            [columns_c])),
        arm_preprocessing_combiner=tf.compat.v2.keras.layers.DenseFeatures(
            [columns_a, columns_b]))
    time_step_spec = ts.time_step_spec(obs_spec)
    reward_layer = get_per_arm_reward_layer(encoding_dim=self._encoding_dim)
    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        dummy_net,
        self._encoding_dim,
        reward_layer,
        actions_from_reward_layer=tf.constant(
            actions_from_reward_layer, dtype=tf.bool),
        cov_matrix=self._a[0:1],
        data_vector=self._b[0:1],
        num_samples=self._num_samples_per_arm[0:1],
        epsilon_greedy=0.0,
        time_step_spec=time_step_spec,
        accepts_per_arm_features=True,
        emit_policy_info=('predicted_rewards_mean',))
    observations = {
        'global': {
            'sport': tf.constant(['snooker', 'chess'])
        },
        'per_arm': {
            'name':
                tf.constant([['george', 'george', 'george'],
                             ['bob', 'bob', 'bob']]),
            'fruit':
                tf.constant([['banana', 'banana', 'banana'],
                             ['kiwi', 'kiwi', 'kiwi']])
        }
    }

    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate([tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.tables_initializer()])
    action = self.evaluate(action_step.action)
    self.assertAllEqual(action.shape, [2])
    p_info = self.evaluate(action_step.info)
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 3])
    self.assertAllEqual(p_info.chosen_arm_features['name'].shape, [2])
    self.assertAllEqual(p_info.chosen_arm_features['fruit'].shape, [2])
    first_action = action[0]
    first_arm_name_feature = observations[
        bandit_spec_utils.PER_ARM_FEATURE_KEY]['name'][0]
    self.assertAllEqual(p_info.chosen_arm_features['name'][0],
                        first_arm_name_feature[first_action])

if __name__ == '__main__':
  tf.test.main()
