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

"""Tests for agents.dqn.dqn_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.networks import sequential
from tf_agents.networks import test_utils as networks_test_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import test_utils as trajectories_test_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import test_utils


class DummyNet(network.Network):

  def __init__(self,
               observation_spec,
               action_spec,
               l2_regularization_weight=0.0,
               name=None):
    super(DummyNet, self).__init__(
        observation_spec, state_spec=(), name=name)
    num_actions = action_spec.maximum - action_spec.minimum + 1

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_regularizer=tf.keras.regularizers.l2(
                l2_regularization_weight),
            kernel_initializer=tf.constant_initializer([[num_actions, 1],
                                                        [1, 1]]),
            bias_initializer=tf.constant_initializer([[1], [1]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class ComputeTDTargetsTest(test_utils.TestCase):

  def testComputeTDTargets(self):
    next_q_values = tf.constant([10, 20], dtype=tf.float32)
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    expected_td_targets = [19., 38.]
    td_targets = dqn_agent.compute_td_targets(next_q_values, rewards, discounts)
    self.assertAllClose(self.evaluate(td_targets), expected_td_targets)


@parameterized.named_parameters(
    ('DqnAgent', dqn_agent.DqnAgent),
    ('DdqnAgent', dqn_agent.DdqnAgent))
class DqnAgentTest(test_utils.TestCase):

  def setUp(self):
    super(DqnAgentTest, self).setUp()
    self._observation_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._observation_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 1)

  def testCreateAgent(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    self.assertIsNotNone(agent.policy)

  def testCreateAgentWithPrebuiltPreprocessingLayers(self, agent_class):
    dense_layer = tf.keras.layers.Dense(2)
    q_net = networks_test_utils.KerasLayersNet(self._observation_spec,
                                               self._action_spec,
                                               dense_layer)
    with self.assertRaisesRegexp(
        ValueError, 'shares weights with the original network'):
      agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None)

    # Explicitly share weights between q and target networks.
    # This would be an unusual setup so we check that an error is thrown.
    q_target_net = networks_test_utils.KerasLayersNet(self._observation_spec,
                                                      self._action_spec,
                                                      dense_layer)
    with self.assertRaisesRegexp(
        ValueError, 'shares weights with the original network'):
      agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None,
          target_q_network=q_target_net)

  def testInitializeAgent(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

  def testCreateAgentDimChecks(self, agent_class):
    action_spec = tensor_spec.BoundedTensorSpec([1, 2], tf.int32, 0, 1)
    q_net = DummyNet(self._observation_spec, action_spec)
    with self.assertRaisesRegex(ValueError, 'Only scalar actions'):
      agent_class(
          self._time_step_spec, action_spec, q_network=q_net, optimizer=None)

  def testInvalidNetworkOutputSize(self, agent_class):
    wrong_action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    q_net = q_network.QNetwork(
        self._time_step_spec.observation,
        wrong_action_spec)
    with self.assertRaisesRegex(ValueError, r'with inner dims \(2,\)'):
      agent_class(
          self._time_step_spec, self._action_spec,
          q_network=q_net, optimizer=None)

  # TODO(b/127383724): Add a test where the target network has different values.
  def testLoss(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    experience = trajectories_test_utils.stacked_trajectory_from_transition(
        time_steps, action_steps, next_time_steps)

    # Using the kernel initializer [[2, 1], [1, 1]] and bias initializer
    # [[1], [1]] from DummyNet above, we can calculate the following values:
    # Q-value for first observation/action pair: 2 * 1 + 1 * 2 + 1 = 5
    # Q-value for second observation/action pair: 1 * 3 + 1 * 4 + 1 = 8
    # (Here we use the second row of the kernel initializer above, since the
    # chosen action is now 1 instead of 0.)
    #
    # For target Q-values, action 0 produces a greater Q-value with a kernel of
    # [2, 1] instead of [1, 1] for action 1.
    # Target Q-value for first next_observation: 2 * 5 + 1 * 6 + 1 = 17
    # Target Q-value for second next_observation: 2 * 7 + 1 * 8 + 1 = 23
    # TD targets: 10 + 0.9 * 17 = 25.3 and 20 + 0.9 * 23 = 40.7
    # TD errors: 25.3 - 5 = 20.3 and 40.7 - 8 = 32.7
    # TD loss: 19.8 and 32.2 (Huber loss subtracts 0.5)
    # Overall loss: (19.8 + 32.2) / 2 = 26
    expected_loss = 26.0
    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossWithChangedOptimalActions(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    # Note that instead of [[5, 6], [7, 8]] as before, we now have -5 and -7.
    next_observations = tf.constant([[-5, 6], [-7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    experience = trajectories_test_utils.stacked_trajectory_from_transition(
        time_steps, action_steps, next_time_steps)

    # Using the kernel initializer [[2, 1], [1, 1]] and bias initializer
    # [[1], [1]] from DummyNet above, we can calculate the following values:
    # Q-value for first observation/action pair: 2 * 1 + 1 * 2 + 1 = 5
    # Q-value for second observation/action pair: 1 * 3 + 1 * 4 + 1 = 8
    # (Here we use the second row of the kernel initializer above, since the
    # chosen action is now 1 instead of 0.)
    #
    # For the target Q-values here, note that since we've replaced 5 and 7 with
    # -5 and -7, it is better to use action 1 with a kernel of [1, 1] instead of
    # action 0 with a kernel of [2, 1].
    # Target Q-value for first next_observation: 1 * -5 + 1 * 6 + 1 = 2
    # Target Q-value for second next_observation: 1 * -7 + 1 * 8 + 1 = 2
    # TD targets: 10 + 0.9 * 2 = 11.8 and 20 + 0.9 * 2 = 21.8
    # TD errors: 11.8 - 5 = 6.8 and 21.8 - 8 = 13.8
    # TD loss: 6.3 and 13.3 (Huber loss subtracts 0.5)
    # Overall loss: (6.3 + 13.3) / 2 = 9.8
    expected_loss = 9.8
    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossWithL2Regularization(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec,
                     l2_regularization_weight=1.0)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    experience = trajectories_test_utils.stacked_trajectory_from_transition(
        time_steps, action_steps, next_time_steps)

    # See the loss explanation in testLoss above.
    # L2_regularization_loss: 2^2 + 1^2 + 1^2 + 1^2 = 7.0
    # Overall loss: 26.0 (from testLoss) + 7.0 = 33.0
    expected_loss = 33.0
    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossNStep(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        n_step_update=2)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    third_observations = tf.constant([[9, 10], [11, 12]], dtype=tf.float32)
    third_time_steps = ts.transition(third_observations, rewards, discounts)

    experience1 = trajectory.from_transition(
        time_steps, action_steps, next_time_steps)
    experience2 = trajectory.from_transition(
        next_time_steps, action_steps, third_time_steps)
    experience3 = trajectory.from_transition(
        third_time_steps, action_steps, third_time_steps)

    experience = tf.nest.map_structure(
        lambda x, y, z: tf.stack([x, y, z], axis=1),
        experience1, experience2, experience3)

    # We can extend the analysis from testLoss above as follows:
    # Original Q-values are still 5 and 8 for the same reasons.
    # Q-value for first third_observation: 2 * 9 + 1 * 10 + 1 = 29
    # Q-value for second third_observation: 2 * 11 + 1 * 12 + 1 = 35
    # TD targets: 10 + 0.9 * (10 + 0.9 * 29) = 42.49
    # 20 + 0.9 * (20 + 0.9 * 35) = 66.35
    # TD errors: 42.49 - 5 = 37.49 and 66.35 - 8 = 58.35
    # TD loss: 36.99 and 57.85 (Huber loss subtracts 0.5)
    # Overall loss: (36.99 + 57.85) / 2 = 47.42
    expected_loss = 47.42
    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossRNNSmokeTest(self, agent_class):
    q_net = sequential.Sequential([
        tf.keras.layers.LSTM(
            2, return_state=True, return_sequences=True,
            kernel_initializer=tf.constant_initializer(0.5),
            recurrent_initializer=tf.constant_initializer(0.5)),
    ])

    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        gamma=0.95,
        optimizer=None)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.7, 0.8], dtype=tf.float32)

    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)
    third_observations = tf.constant([[9, 10], [11, 12]], dtype=tf.float32)
    third_time_steps = ts.transition(third_observations, rewards, discounts)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    experience1 = trajectory.from_transition(
        time_steps, action_steps, next_time_steps)
    experience2 = trajectory.from_transition(
        next_time_steps, action_steps, third_time_steps)
    experience3 = trajectory.from_transition(
        third_time_steps, action_steps, third_time_steps)

    experience = tf.nest.map_structure(
        lambda x, y, z: tf.stack([x, y, z], axis=1),
        experience1, experience2, experience3)

    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Smoke test, here to make sure the calculation does not change as we
    # modify preprocessing or other internals.
    expected_loss = 28.722265
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossNStepMidMidLastFirst(self, agent_class):
    """Tests that n-step loss handles LAST time steps properly."""
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        n_step_update=3)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    # MID: use ts.transition
    time_steps = ts.transition(observations, rewards, discounts)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    second_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    # MID: use ts.transition
    second_time_steps = ts.transition(second_observations, rewards, discounts)

    third_observations = tf.constant([[9, 10], [11, 12]], dtype=tf.float32)
    # LAST: use ts.termination
    third_time_steps = ts.termination(third_observations, rewards)

    fourth_observations = tf.constant([[13, 14], [15, 16]], dtype=tf.float32)
    # FIRST: use ts.restart
    fourth_time_steps = ts.restart(fourth_observations, batch_size=2)

    experience1 = trajectory.from_transition(
        time_steps, action_steps, second_time_steps)
    experience2 = trajectory.from_transition(
        second_time_steps, action_steps, third_time_steps)
    experience3 = trajectory.from_transition(
        third_time_steps, action_steps, fourth_time_steps)
    experience4 = trajectory.from_transition(
        fourth_time_steps, action_steps, fourth_time_steps)

    experience = tf.nest.map_structure(
        lambda w, x, y, z: tf.stack([w, x, y, z], axis=1),
        experience1, experience2, experience3, experience4)

    # Once again we can extend the analysis from testLoss above as follows:
    # Original Q-values are still 5 and 8 for the same reasons.
    # However next Q-values are now zeroed out due to the LAST time step in
    # between. Thus the TD targets become the discounted reward sums, or:
    # 10 + 0.9 * 10 = 19 and 20 + 0.9 * 20 = 38
    # TD errors: 19 - 5 = 14 and 38 - 8 = 30
    # TD loss: 13.5 and 29.5 (Huber loss subtracts 0.5)
    # Overall loss: (13.5 + 29.5) / 2 = 21.5
    expected_loss = 21.5
    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossWithMaskedActions(self, agent_class):
    # Observations are now a tuple of the usual observation and an action mask.
    observation_spec_with_mask = (
        self._observation_spec,
        tensor_spec.BoundedTensorSpec([2], tf.int32, 0, 1))
    time_step_spec = ts.time_step_spec(observation_spec_with_mask)
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]))

    # For `observations`, the masks are set up so that all actions are valid.
    observations = (tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
                    tf.constant([[1, 1], [1, 1]], dtype=tf.int32))
    time_steps = ts.restart(observations, batch_size=2)

    actions = tf.constant([0, 1], dtype=tf.int32)
    action_steps = policy_step.PolicyStep(actions)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    # For `next_observations`, the masks are set up so that only one action is
    # valid for each element in the batch.
    next_observations = (tf.constant([[5, 6], [7, 8]], dtype=tf.float32),
                         tf.constant([[0, 1], [1, 0]], dtype=tf.int32))
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    experience = trajectories_test_utils.stacked_trajectory_from_transition(
        time_steps, action_steps, next_time_steps)

    # Using the kernel initializer [[2, 1], [1, 1]] and bias initializer
    # [[1], [1]] from DummyNet above, we can calculate the following values:
    # Q-value for first observation/action pair: 2 * 1 + 1 * 2 + 1 = 5
    # Q-value for second observation/action pair: 1 * 3 + 1 * 4 + 1 = 8
    # (Here we use the second row of the kernel initializer above, since the
    # chosen action is now 1 instead of 0.)
    #
    # For target Q-values, because of the masks we only have one valid choice of
    # action for each next_observation:
    # Target Q-value for first next_observation (only action 1 is valid):
    # 1 * 5 + 1 * 6 + 1 = 12
    # Target Q-value for second next_observation (only action 0 is valid):
    # 2 * 7 + 1 * 8 + 1 = 23
    # TD targets: 10 + 0.9 * 12 = 20.8 and 20 + 0.9 * 23 = 40.7
    # TD errors: 20.8 - 5 = 15.8 and 40.7 - 8 = 32.7
    # TD loss: 15.3 and 32.2 (Huber loss subtracts 0.5)
    # Overall loss: (15.3 + 32.2) / 2 = 23.75
    expected_loss = 23.75
    loss, _ = agent._loss(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  def testPolicy(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual(action_step.action.shape,
                        [2] + self._action_spec.shape.as_list())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(all(actions_ <= self._action_spec.maximum))
    self.assertTrue(all(actions_ >= self._action_spec.minimum))

  def testInitializeRestoreAgent(self, agent_class):
    q_net = DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    checkpoint = tf.train.Checkpoint(agent=agent)

    latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)

    if tf.executing_eagerly():
      self.evaluate(checkpoint_load_status.initialize_or_restore())
      self.assertAllEqual(self.evaluate(action_step.action), [0, 0])
    else:
      with self.cached_session() as sess:
        checkpoint_load_status.initialize_or_restore(sess)
        self.assertAllEqual(sess.run(action_step.action), [0, 0])

  def testTrainWithSparseTensorAndDenseFeaturesLayer(self, agent_class):
    obs_spec = {
        'dense': tensor_spec.BoundedTensorSpec(
            dtype=tf.float32, shape=[3], minimum=-10.0, maximum=10.0),
        'sparse_terms': tf.SparseTensorSpec(dtype=tf.string, shape=[4]),
        'sparse_frequencies': tf.SparseTensorSpec(dtype=tf.float32, shape=[4]),
    }
    cat_column = (
        tf.compat.v2.feature_column.categorical_column_with_hash_bucket(
            'sparse_terms', hash_bucket_size=5))
    weighted_cat_column = (
        tf.compat.v2.feature_column.weighted_categorical_column(
            cat_column, weight_feature_key='sparse_frequencies'))
    feature_columns = [
        tf.compat.v2.feature_column.numeric_column('dense', [3]),
        tf.compat.v2.feature_column.embedding_column(weighted_cat_column, 3),
    ]
    dense_features_layer = tf.compat.v2.keras.layers.DenseFeatures(
        feature_columns)
    time_step_spec = ts.time_step_spec(obs_spec)
    q_net = q_network.QNetwork(
        time_step_spec.observation,
        self._action_spec,
        preprocessing_combiner=dense_features_layer)
    agent = agent_class(
        time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=tf.compat.v1.train.AdamOptimizer())

    observations = tensor_spec.sample_spec_nest(obs_spec, outer_dims=[5, 2])
    # sparse_terms and sparse_frequencies must be defined on matching indices.
    observations['sparse_terms'] = tf.SparseTensor(
        indices=observations['sparse_frequencies'].indices,
        values=tf.as_string(
            tf.math.round(observations['sparse_frequencies'].values)),
        dense_shape=observations['sparse_frequencies'].dense_shape)
    if not tf.executing_eagerly():
      # Mimic unknown inner dims on the SparseTensor
      def _unknown_inner_shape(t):
        if not isinstance(t, tf.SparseTensor):
          return t
        return tf.SparseTensor(
            indices=t.indices, values=t.values,
            dense_shape=tf.compat.v1.placeholder_with_default(
                t.dense_shape, shape=t.dense_shape.shape))
      observations = tf.nest.map_structure(_unknown_inner_shape, observations)
      self.assertIsNone(tf.get_static_value(
          observations['sparse_terms'].dense_shape))

    time_step = ts.restart(observations, batch_size=[5, 2])
    action_step = tensor_spec.sample_spec_nest(
        self._action_spec, outer_dims=[5, 2])
    p_step = policy_step.PolicyStep(action=action_step, state=(), info=())
    traj = trajectory.from_transition(time_step, p_step, time_step)
    loss_info = agent.train(traj)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss_info)
    self.assertGreater(loss_info.loss, 0)


if __name__ == '__main__':
  tf.test.main()
