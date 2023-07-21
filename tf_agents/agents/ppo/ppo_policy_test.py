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

"""Tests for tf_agents.policies.actor_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_policy
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import mask_splitter_network
from tf_agents.networks import network
from tf_agents.networks import sequential
from tf_agents.networks import value_network as value_net
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

tfd = tfp.distributions


class DummyActorNet(network.Network):

  def __init__(self, action_spec, emit_distribution=True, name=None):
    super(DummyActorNet, self).__init__(
        tensor_spec.TensorSpec([2], tf.float32), (), 'DummyActorNet')
    self._action_spec = action_spec
    self._flat_action_spec = tf.nest.flatten(self._action_spec)[0]

    self._dummy_layers = [
        tf.keras.layers.Dense(
            self._flat_action_spec.shape.num_elements(),
            kernel_initializer=tf.constant_initializer([2, 1]),
            bias_initializer=tf.constant_initializer([5]),
            activation=tf.keras.activations.tanh,
        )
    ]

    self._emit_distribution = emit_distribution

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]
    for layer in self._dummy_layers:
      hidden_state = layer(hidden_state)

    means = tf.reshape(hidden_state,
                       [-1] + self._flat_action_spec.shape.as_list())
    spec_means = (
        self._flat_action_spec.maximum + self._flat_action_spec.minimum) / 2.0
    spec_ranges = (
        self._flat_action_spec.maximum - self._flat_action_spec.minimum) / 2.0
    action_means = spec_means + spec_ranges * means
    if self._emit_distribution:
      action_means = tfd.Deterministic(loc=action_means)

    return tf.nest.pack_sequence_as(self._action_spec,
                                    [action_means]), network_state


class DummyActorDistributionNet(network.DistributionNetwork):

  def __init__(self, action_spec, name=None):
    output_spec = tf.nest.map_structure(self._get_normal_distribution_spec,
                                        action_spec)
    super(DummyActorDistributionNet, self).__init__(
        tensor_spec.TensorSpec([2], tf.float32),
        (),
        output_spec=output_spec,
        name='DummyActorDistributionNet')
    self._action_net = DummyActorNet(action_spec, emit_distribution=False)

  def _get_normal_distribution_spec(self, sample_spec):
    param_properties = tfp.distributions.Normal.parameter_properties()
    input_param_spec = {  # pylint: disable=g-complex-comprehension
        name: tensor_spec.TensorSpec(
            shape=properties.shape_fn(sample_spec.shape),
            dtype=sample_spec.dtype)
        for name, properties in param_properties.items()
    }

    return distribution_spec.DistributionSpec(
        tfp.distributions.Normal, input_param_spec, sample_spec=sample_spec)

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    action_means, network_state = self._action_net(inputs, network_state)

    def _action_distribution(action_mean):
      action_std = tf.ones_like(action_mean)
      return tfp.distributions.Normal(action_mean, action_std)

    return tf.nest.map_structure(_action_distribution,
                                 action_means), network_state


class DummyValueNet(network.Network):

  def __init__(self, name=None):
    super(DummyValueNet, self).__init__(
        tensor_spec.TensorSpec([2], tf.float32), (), 'DummyValueNet')
    self._dummy_layers = [
        tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.constant_initializer([2, 1]),
            bias_initializer=tf.constant_initializer([5]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]
    for layer in self._dummy_layers:
      hidden_state = layer(hidden_state)
    return hidden_state, network_state


def create_sequential_actor_net():
  def create_dist(loc_and_scale):
    # Bring my_action into [2.0, 3.0]:
    #  (-inf, inf) -> (-1, 1) -> (-0.5, 0.5) -> (2, 3)
    my_action = tfp.bijectors.Chain(
        [tfp.bijectors.Shift(2.5), tfp.bijectors.Scale(0.5),
         tfp.bijectors.Tanh()])(
             tfd.Normal(
                 loc=loc_and_scale[..., 0],
                 scale=tf.math.softplus(loc_and_scale[..., 1]) + 1e-10,
                 validate_args=True))
    return {
        'my_action': my_action,
    }

  tf.keras.utils.set_random_seed(0)
  return sequential.Sequential([
      tf.keras.layers.Dense(4),
      tf.keras.layers.Dense(2),
      tf.keras.layers.Lambda(create_dist)
  ])


def _test_cases(prefix=''):
  return [{
      'testcase_name': '%s0' % prefix,
      'network_cls': DummyActorNet,
  }, {
      'testcase_name': '%s1' % prefix,
      'network_cls': DummyActorDistributionNet,
  }]


class PPOPolicyTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(PPOPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.float32, 2, 3)

  @property
  def _time_step(self):
    return ts.TimeStep(
        step_type=tf.constant([1], dtype=tf.int32),
        reward=tf.constant([1], dtype=tf.float32),
        discount=tf.constant([1], dtype=tf.float32),
        observation=tf.constant([[1, 2]], dtype=tf.float32))

  @property
  def _time_step_batch(self):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, dtype=tf.int32, shape=[2], name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[2], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[2], name='discount'),
        tf.constant([[1, 2], [3, 4]], dtype=tf.float32, name='observation'))

  @parameterized.named_parameters(*_test_cases('test_build'))
  def testBuild(self, network_cls):
    actor_network = network_cls(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  @parameterized.named_parameters(*_test_cases('test_reset'))
  def testReset(self, network_cls):
    actor_network = network_cls(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    policy_state = policy.get_initial_state(batch_size=1)

    # Dummy network has no policy_state so expect empty tuple from reset.
    self.assertEqual((), policy_state)

  @parameterized.named_parameters(*_test_cases('test_action'))
  def testAction(self, network_cls):
    actor_network = network_cls(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    self.assertEqual(action_step.action.shape.as_list(), [1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  @parameterized.named_parameters(*_test_cases('test_action'))
  def testValueInPolicyInfo(self, network_cls):
    actor_network = network_cls(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    policy_step = policy.action(self._time_step)
    self.assertEqual(policy_step.info['value_prediction'].shape.as_list(),
                     [1, 1])
    self.assertEqual(policy_step.info['value_prediction'].dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(policy_step.info['value_prediction'])

  @parameterized.named_parameters(*_test_cases('test_action_list'))
  def testActionList(self, network_cls):
    action_spec = [self._action_spec]
    actor_network = network_cls(action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        action_spec,
        actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    self.assertIsInstance(action_step.action, list)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= action_spec[0].minimum))
    self.assertTrue(np.all(actions_ <= action_spec[0].maximum))

  @parameterized.named_parameters(*_test_cases('test_action_batch'))
  def testActionBatch(self, network_cls):
    actor_network = network_cls(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  def testPolicyStepWithActionMaskTurnedOn(self):
    # Creat specs with action constraints (mask).
    num_categories = 5
    observation_tensor_spec = (
        tensor_spec.TensorSpec(shape=(3,), dtype=tf.int64, name='network_spec'),
        tensor_spec.TensorSpec(
            shape=(num_categories,), dtype=tf.bool, name='mask_spec'),
    )
    network_spec, _ = observation_tensor_spec
    action_tensor_spec = tensor_spec.BoundedTensorSpec(
        [], tf.int32, 0, num_categories - 1)

    # Create policy with splitter.
    def splitter_fn(observation_and_mask):
      return observation_and_mask[0], observation_and_mask[1]

    actor_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn,
        actor_distribution_network.ActorDistributionNetwork(
            network_spec, action_tensor_spec),
        passthrough_mask=True)
    value_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn, value_net.ValueNetwork(network_spec))
    policy = ppo_policy.PPOPolicy(
        ts.time_step_spec(observation_tensor_spec),
        action_tensor_spec,
        actor_network=actor_network,
        value_network=value_network,
        clip=False)

    # Take a step.
    mask = np.array([True, False, True, False, True], dtype=bool)
    self.assertLen(mask, num_categories)
    time_step = ts.TimeStep(
        step_type=tf.constant([1], dtype=tf.int32),
        reward=tf.constant([1], dtype=tf.float32),
        discount=tf.constant([1], dtype=tf.float32),
        observation=(tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64),
                     tf.constant([mask.tolist()], dtype=tf.bool)))
    action_step = policy.action(time_step)

    # Check the shape and type of the resulted action step.
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Check the actions in general and with respect to masking.
    actions = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions >= action_tensor_spec.minimum))
    self.assertTrue(np.all(actions <= action_tensor_spec.maximum))

    # Check the logits.
    logits = np.array(
        self.evaluate(action_step.info['dist_params']['logits']),
        dtype=np.float32)
    masked_actions = np.array(range(len(mask)))[~mask]
    self.assertTrue(
        np.all(logits[:, masked_actions] == np.finfo(np.float32).min))
    valid_actions = np.array(range(len(mask)))[mask]
    self.assertTrue(
        np.all(logits[:, valid_actions] > np.finfo(np.float32).min))

  @parameterized.named_parameters(*_test_cases('test_action'))
  def testValue(self, network_cls):
    actor_network = network_cls(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    batch_size = tf.compat.dimension_value(self._time_step.step_type.shape[0])
    policy_state = policy.get_initial_state(batch_size=batch_size)
    value_pred, unused_policy_state = policy.apply_value_network(
        self._time_step.observation, self._time_step.step_type, policy_state)
    self.assertEqual(value_pred.shape.as_list(), [1, 1])
    self.assertEqual(value_pred.dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(value_pred)

  def testUpdate(self):
    tf.compat.v1.set_random_seed(1)
    actor_network = DummyActorNet(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)
    new_policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    new_action_step = new_policy.action(self._time_step)

    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(new_policy.update(policy))
    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  def testDeterministicDistribution(self):
    actor_network = DummyActorNet(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    distribution_step = policy.distribution(self._time_step)
    self.assertIsInstance(distribution_step.action,
                          tfp.distributions.Deterministic)
    distribution_mean = distribution_step.action.mean()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    distribution_mean_ = self.evaluate(distribution_mean)
    self.assertNear(actions_, distribution_mean_, 1e-6)

  def testGaussianDistribution(self):
    actor_network = DummyActorDistributionNet(self._action_spec)
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_network,
        value_network=value_network)

    distribution_step = policy.distribution(self._time_step)
    self.assertIsInstance(distribution_step.action, tfp.distributions.Normal)

  def testNonLegacyDistribution(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping test: sequential networks not supported in TF1')

    actor_network = create_sequential_actor_net()
    action_spec = {'my_action': self._action_spec}
    value_network = DummyValueNet()

    policy = ppo_policy.PPOPolicy(
        self._time_step_spec,
        action_spec,
        actor_network=actor_network,
        value_network=value_network)

    distribution_step = policy.distribution(self._time_step)
    self.assertIsInstance(
        distribution_step.action['my_action'],
        tfp.distributions.TransformedDistribution)

    expected_info_spec = {
        'dist_params': {
            'my_action': {
                'bijector': {'bijectors:0': {},
                             'bijectors:1': {},
                             'bijectors:2': {}},
                'distribution': {'scale': tf.TensorSpec([1], tf.float32),
                                 'loc': tf.TensorSpec([1], tf.float32)},
            }
        },
        'value_prediction': tf.TensorSpec([1, 1], tf.float32)
    }

    tf.nest.map_structure(
        lambda v, s: self.assertEqual(tf.type_spec_from_value(v), s),
        distribution_step.info, expected_info_spec)


if __name__ == '__main__':
  tf.test.main()
