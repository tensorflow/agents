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

"""Tests for tf_agents.bandits.agents.constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


tf.compat.v1.enable_v2_behavior()


class SimpleInputNetworkConstraint(constraints.InputNetworkConstraint):

  def __call__(self, observation, actions=None):
    return tf.ones([2, 3])


class SimpleConstraint(constraints.BaseConstraint):

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    batch_size = tf.shape(observation)[0]
    num_actions = self._action_spec.maximum - self._action_spec.minimum + 1
    feasibility_prob = 0.5 * tf.ones([batch_size, num_actions], tf.float32)
    return feasibility_prob


class GreaterThan2Constraint(constraints.BaseConstraint):

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    if actions is None:
      actions = tf.range(self._action_spec.minimum, self._action_spec.maximum)
    feasibility_prob = tf.cast(tf.greater(actions, 2), tf.float32)
    return feasibility_prob


class BaseConstraintTest(tf.test.TestCase):

  def testSimpleCase(self):
    obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=5)
    gt2c = GreaterThan2Constraint(time_step_spec, action_spec)
    feasibility_prob = gt2c(observation=None)
    self.assertAllEqual([0, 0, 0, 1, 1], self.evaluate(feasibility_prob))


class DummyNet(network.Network):

  def __init__(self, unused_observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(
        unused_observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer([[1, 1.5, 2],
                                                        [1, 1.5, 4]]),
            bias_initializer=tf.constant_initializer([[1], [1], [-10]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class NeuralConstraintTest(tf.test.TestCase):

  def setUp(self):
    super(NeuralConstraintTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation

  def testCreateConstraint(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

  def testInitializeConstraint(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    neural_constraint = constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)
    init_op = neural_constraint.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

  def testComputeLoss(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    actions = tf.constant([0, 1], dtype=tf.int32)
    rewards = tf.constant([0.5, 3.0], dtype=tf.float32)

    neural_constraint = constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)
    init_op = neural_constraint.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))
    loss = neural_constraint.compute_loss(
        observations,
        actions,
        rewards)
    self.assertAllClose(self.evaluate(loss), 42.25)

  def testComputeLossWithArmFeatures(self):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        global_dim=2, per_arm_dim=3, max_num_actions=3)
    time_step_spec = ts.time_step_spec(obs_spec)
    constraint_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec,
            global_layers=(4,),
            arm_layers=(4,),
            common_layers=(4,)))
    neural_constraint = constraints.NeuralConstraint(
        time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tf.cast(
                tf.reshape(tf.range(18), shape=[2, 3, 3]), dtype=tf.float32)
    }
    actions = tf.constant([0, 1], dtype=tf.int32)
    rewards = tf.constant([0.5, 3.0], dtype=tf.float32)

    init_op = neural_constraint.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))
    loss = neural_constraint.compute_loss(
        observations,
        actions,
        rewards)
    self.assertGreater(self.evaluate(loss), 0.0)

  def testComputeActionFeasibility(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)

    neural_constraint = constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)
    init_op = neural_constraint.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = neural_constraint(observation)
    self.assertAllClose(self.evaluate(feasibility_prob), np.ones([2, 3]))


class RelativeConstraintTest(tf.test.TestCase):

  def setUp(self):
    super(RelativeConstraintTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)

  def testComputeActionFeasibilityNoBaselineActionFn(self):
    constraint_net = DummyNet(self._obs_spec, self._action_spec)
    relative_constraint = constraints.RelativeConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net,
        baseline_action_fn=None)
    init_op = relative_constraint.initialize()
    self.evaluate(init_op)

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = relative_constraint(observation)
    self.assertAllEqual(self.evaluate(feasibility_prob),
                        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))

  def testComputeActionFeasibility(self):
    constraint_net = DummyNet(self._obs_spec, self._action_spec)
    baseline_action_fn = lambda _: tf.constant([1, 1], dtype=tf.int32)
    relative_constraint = constraints.RelativeConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net,
        baseline_action_fn=baseline_action_fn)
    init_op = relative_constraint.initialize()
    self.evaluate(init_op)

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = relative_constraint(observation)
    self.assertAllEqual(self.evaluate(feasibility_prob),
                        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


class AbsoluteConstraintTest(tf.test.TestCase):

  def setUp(self):
    super(AbsoluteConstraintTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation

  def testCreateConstraint(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    constraints.AbsoluteConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

  def testComputeActionFeasibility(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)

    absolute_constraint = constraints.AbsoluteConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)
    init_op = absolute_constraint.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = absolute_constraint(observation)
    self.assertAllGreaterEqual(self.evaluate(feasibility_prob), 0.0)
    self.assertAllLessEqual(self.evaluate(feasibility_prob), 1.0)


class QuantileConstraintTest(tf.test.TestCase):

  def setUp(self):
    super(QuantileConstraintTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation

  def testCreateConstraint(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    constraints.QuantileConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

  def testComputeActionFeasibility(self):
    constraint_net = DummyNet(self._observation_spec, self._action_spec)

    quantile_constraint = constraints.QuantileConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)
    init_op = quantile_constraint.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = quantile_constraint(observation)
    self.assertAllGreaterEqual(self.evaluate(feasibility_prob), 0.0)
    self.assertAllLessEqual(self.evaluate(feasibility_prob), 1.0)


class RelativeQuantileConstraintTest(tf.test.TestCase):

  def setUp(self):
    super(RelativeQuantileConstraintTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)

  def testComputeActionFeasibilityNoBaselineActionFn(self):
    constraint_net = DummyNet(self._obs_spec, self._action_spec)
    quantile_constraint = constraints.RelativeQuantileConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net,
        baseline_action_fn=None)
    init_op = quantile_constraint.initialize()
    self.evaluate(init_op)

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = quantile_constraint(observation)
    self.assertAllEqual(self.evaluate(feasibility_prob),
                        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))

  def testComputeActionFeasibility(self):
    constraint_net = DummyNet(self._obs_spec, self._action_spec)
    baseline_action_fn = lambda _: tf.constant([1, 1], dtype=tf.int32)
    quantile_constraint = constraints.RelativeQuantileConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net,
        baseline_action_fn=baseline_action_fn)
    init_op = quantile_constraint.initialize()
    self.evaluate(init_op)

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = quantile_constraint(observation)
    self.assertAllEqual(self.evaluate(feasibility_prob),
                        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


class ConstraintFeasibilityTest(tf.test.TestCase):

  def testComputeFeasibilityMask(self):
    observation_spec = tensor_spec.TensorSpec([2], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    simple_constraint = SimpleConstraint(time_step_spec, action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = constraints.compute_feasibility_probability(
        observations, [simple_constraint], batch_size=2, num_actions=3,
        action_mask=None)
    self.assertAllEqual(0.5 * np.ones([2, 3]), self.evaluate(feasibility_prob))

  def testComputeFeasibilityMaskWithActionMask(self):
    observation_spec = tensor_spec.TensorSpec([2], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    constraint_net = DummyNet(observation_spec, action_spec)
    neural_constraint = constraints.NeuralConstraint(
        time_step_spec,
        action_spec,
        constraint_network=constraint_net)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    action_mask = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
    feasibility_prob = constraints.compute_feasibility_probability(
        observations, [neural_constraint], batch_size=2, num_actions=3,
        action_mask=action_mask)
    self.assertAllEqual(self.evaluate(tf.cast(action_mask, tf.float32)),
                        self.evaluate(feasibility_prob))

  def testComputeMaskFromMultipleSourcesNumActionsFeature(self):
    observation_spec = bandit_spec_utils.create_per_arm_observation_spec(
        4, 5, 6, add_num_actions_feature=True)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 5)
    constraint_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            observation_spec, (3, 4), (4, 3), (2, 3)))
    neural_constraint = constraints.NeuralConstraint(
        time_step_spec,
        action_spec,
        constraint_network=constraint_net)

    observations = {
        'global': tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32),
        'per_arm': tf.reshape(tf.range(60, dtype=tf.float32), shape=[2, 6, 5]),
        'num_actions': tf.constant([4, 3], dtype=tf.int32)
    }
    mask = constraints.construct_mask_from_multiple_sources(
        observations, None, [neural_constraint], 6)
    implied_mask = [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]]
    self.assertAllGreaterEqual(implied_mask - mask, 0)

  def testComputeMaskFromMultipleSourcesMask(self):
    observation_spec = bandit_spec_utils.create_per_arm_observation_spec(
        4, 5, 6)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 5)
    constraint_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            observation_spec, (3, 4), (4, 3), (2, 3)))
    neural_constraint = constraints.NeuralConstraint(
        time_step_spec,
        action_spec,
        constraint_network=constraint_net)
    original_mask = [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]]
    observations = ({
        'global': tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32),
        'per_arm': tf.reshape(tf.range(60, dtype=tf.float32), shape=[2, 6, 5]),
    }, original_mask)
    mask = constraints.construct_mask_from_multiple_sources(
        observations, lambda x: (x[0], x[1]), [neural_constraint], 6)
    self.assertAllGreaterEqual(original_mask - mask, 0)


class InputNetworkConstraintTest(tf.test.TestCase):

  def setUp(self):
    super(InputNetworkConstraintTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation

  def testCreateConstraint(self):
    input_net = DummyNet(self._observation_spec, self._action_spec)
    SimpleInputNetworkConstraint(
        self._time_step_spec,
        self._action_spec,
        input_network=input_net)

  def testComputeLoss(self):
    input_net = DummyNet(self._observation_spec, self._action_spec)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    actions = tf.constant([0, 1], dtype=tf.int32)
    rewards = tf.constant([0.5, 3.0], dtype=tf.float32)

    neural_constraint = SimpleInputNetworkConstraint(
        self._time_step_spec,
        self._action_spec,
        input_network=input_net)
    loss = neural_constraint.compute_loss(
        observations,
        actions,
        rewards)
    self.assertAllClose(self.evaluate(loss), 0.0)

  def testComputeActionFeasibility(self):
    input_net = DummyNet(self._observation_spec, self._action_spec)

    neural_constraint = SimpleInputNetworkConstraint(
        self._time_step_spec,
        self._action_spec,
        input_network=input_net)

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    feasibility_prob = neural_constraint(observation)
    self.assertAllClose(self.evaluate(feasibility_prob), np.ones([2, 3]))


if __name__ == '__main__':
  tf.test.main()
