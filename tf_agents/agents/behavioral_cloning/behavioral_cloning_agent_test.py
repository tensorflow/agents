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

"""Tests for agents.behavioral_cloning.behavioral_cloning_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import test_util
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import expand_dims_layer
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import test_utils

# Number of times to train in test loops.
TRAIN_ITERATIONS = 10


def create_arbitrary_trajectory():
  """Creates an arbitrary trajectory for unit testing BehavioralCloningAgent.

  This trajectory contains Tensors shaped `[6, 1, ...]` where `6` is the number
  of time steps and `1` is the batch.

  Observations are unbounded but actions are bounded to take values within
  `[1, 2]`. The action space is discrete.

  Policy info is not provided, as it is not used in BehavioralCloningAgent.

  Returns:
    traj: a hard coded `Trajectory` that matches time_step_spec and action_spec.
    time_step_spec: a hard coded time spec.
    action_spec: a hard coded action spec.
  """

  time_step_spec = ts.time_step_spec(
      tensor_spec.TensorSpec([], tf.int32, name='observation'))
  action_spec = tensor_spec.BoundedTensorSpec([],
                                              tf.int32,
                                              minimum=1,
                                              maximum=2,
                                              name='action')

  observation = tf.constant([[1], [2], [3], [4], [5], [6]], dtype=tf.int32)
  action = tf.constant([[1], [1], [1], [2], [2], [2]], dtype=tf.int32)
  reward = tf.constant([[0], [0], [0], [0], [0], [0]], dtype=tf.float32)
  step_type = tf.constant([[0], [1], [2], [0], [1], [2]], dtype=tf.int32)
  next_step_type = tf.constant([[1], [2], [0], [1], [2], [0]], dtype=tf.int32)
  discount = tf.constant([[1], [1], [1], [1], [1], [1]], dtype=tf.float32)

  traj = trajectory.Trajectory(
      observation=observation,
      action=action,
      policy_info=(),
      reward=reward,
      step_type=step_type,
      next_step_type=next_step_type,
      discount=discount,
  )
  return traj, time_step_spec, action_spec


def get_dummy_net(action_spec, observation_spec=None):
  flat_action_spec = tf.nest.flatten(action_spec)[0]
  if flat_action_spec.dtype.is_integer:
    # Emitting discrete actions.
    num_actions = flat_action_spec.maximum - flat_action_spec.minimum + 1
    kernel_initializer = tf.constant_initializer([[2, 1], [1, 1]])
    bias_initializer = tf.constant_initializer([[1], [1]])
    final_shape = [num_actions]
  else:
    # Emitting continuous vectors.
    num_actions = np.prod(flat_action_spec.shape)
    kernel_initializer = None
    bias_initializer = None
    final_shape = flat_action_spec.shape

  return sequential.Sequential([
      tf.keras.layers.Dense(
          num_actions,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          dtype=tf.float32),
      inner_reshape.InnerReshape([None], final_shape)
  ], input_spec=observation_spec)


def get_mock_hybrid_loss(actor_net, action_spec):
  def hybrid_loss(experience, training=False):
    del training
    batch_size = (
        tf.compat.dimension_value(experience.step_type.shape[0]) or
        tf.shape(experience.step_type)[0])
    network_state = actor_net.get_initial_state(batch_size)
    # actor may define random ops like cropping. Pass training=False to disable.
    bc_output, _ = actor_net(
        experience.observation,
        step_type=experience.step_type,
        training=False,
        network_state=network_state)
    def _compute_loss(dist, label, spec):
      prediction = dist.mean()
      if spec.dtype.is_integer:
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        return cross_entropy(label, prediction)
      else:
        return tf.reduce_sum(
            tf.math.squared_difference(label, prediction), axis=-1)
    losses_dict = tf.nest.map_structure(
        _compute_loss, bc_output, experience.action,
        action_spec)
    return tf.add_n(tf.nest.flatten(losses_dict))
  return hybrid_loss


class BehavioralCloningAgentTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(BehavioralCloningAgentTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)
    self._observation_spec = self._time_step_spec.observation

  def testCreateAgent(self):
    cloning_net = get_dummy_net(self._action_spec, self._observation_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)
    self.assertIsNotNone(agent.policy)

  @parameterized.named_parameters(
      ('MultipleActions', [
          tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1),
          tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)],
       '.* single, scalar discrete.*'),
      ('NonScalarAction', [
          tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)],
       '.* single, scalar discrete.*'),
      ('ScalarDiscreteAction', [
          tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)],
       None),
      ('SingleNonScalarFloatAction',
       tensor_spec.BoundedTensorSpec([3, 2], tf.float32, 0, 1),
       None),
      ('MultipleContinuous', [
          tensor_spec.BoundedTensorSpec([], tf.float32, 0, 1),
          tensor_spec.BoundedTensorSpec([], tf.float32, 0, 1)],
       '.* single, scalar discrete.*'),
      ('MixedDiscreteAndContinuous', [
          tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1),
          tensor_spec.BoundedTensorSpec([], tf.float32, 0, 1)],
       '.* single, scalar discrete.*'))
  def testCreateAgentNestSizeChecks(self,
                                    action_spec,
                                    expected_error):
    cloning_net = get_dummy_net(action_spec, self._observation_spec)
    if expected_error is not None:
      with self.assertRaisesRegex(ValueError, expected_error):
        behavioral_cloning_agent.BehavioralCloningAgent(
            self._time_step_spec,
            action_spec,
            cloning_network=cloning_net,
            optimizer=None)
    else:
      behavioral_cloning_agent.BehavioralCloningAgent(
          self._time_step_spec,
          action_spec,
          cloning_network=cloning_net,
          optimizer=None)

  def verifyVariableAssignAndRestore(self,
                                     observation_spec,
                                     action_spec,
                                     actor_net,
                                     loss_fn=None):
    strategy = tf.distribute.get_strategy()
    time_step_spec = ts.time_step_spec(observation_spec)
    with strategy.scope():
      # Use BehaviorCloningAgent instead of AWRAgent to test the network.
      agent = behavioral_cloning_agent.BehavioralCloningAgent(
          time_step_spec,
          action_spec,
          cloning_network=actor_net,
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
          loss_fn=loss_fn)
    # Assign all vars to 0.
    for var in tf.nest.flatten(agent.variables):
      var.assign(tf.zeros_like(var))
    # Save checkpoint
    ckpt_dir = self.create_tempdir()
    checkpointer = common.Checkpointer(
        ckpt_dir=ckpt_dir, agent=agent)
    global_step = tf.constant(0)
    checkpointer.save(global_step)
    # Assign all vars to 1.
    for var in tf.nest.flatten(agent.variables):
      var.assign(tf.ones_like(var))
    # Restore to 0.
    checkpointer._checkpoint.restore(checkpointer._manager.latest_checkpoint)
    for var in tf.nest.flatten(agent.variables):
      value = var.numpy()
      if isinstance(value, np.int64):
        self.assertEqual(value, 0)
      else:
        self.assertAllEqual(
            value, np.zeros_like(value),
            msg='{} has var mean {}, expected 0.'.format(var.name, value))

  def verifyTrainAndRestore(self,
                            observation_spec,
                            action_spec,
                            actor_net,
                            loss_fn=None):
    """Helper function for testing correct variable updating and restoring."""
    batch_size = 2
    observations = tensor_spec.sample_spec_nest(
        observation_spec, outer_dims=(batch_size,))
    actions = tensor_spec.sample_spec_nest(
        action_spec, outer_dims=(batch_size,))
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    experience = trajectory.first(
        observation=observations,
        action=actions,
        policy_info=(),
        reward=rewards,
        discount=discounts)
    time_step_spec = ts.time_step_spec(observation_spec)
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
      # Use BehaviorCloningAgent instead of AWRAgent to test the network.
      agent = behavioral_cloning_agent.BehavioralCloningAgent(
          time_step_spec,
          action_spec,
          cloning_network=actor_net,
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
          loss_fn=loss_fn)
    loss_before_train = agent.loss(experience).loss
    # Check loss is stable.
    self.assertEqual(loss_before_train, agent.loss(experience).loss)
    # Train 2 steps, verify that loss is decreased for the same input.
    for _ in range(2):
      agent.train(experience)
    loss_after_train = agent.loss(experience).loss
    self.assertLessEqual(loss_after_train, loss_before_train)
    # Assert loss evaluation is still stable, e.g. deterministic.
    self.assertLessEqual(loss_after_train, agent.loss(experience).loss)
    # Save checkpoint
    ckpt_dir = self.create_tempdir()
    checkpointer = common.Checkpointer(ckpt_dir=ckpt_dir, agent=agent)
    global_step = tf.constant(1)
    checkpointer.save(global_step)
    # Assign all vars to 0.
    for var in tf.nest.flatten(agent.variables):
      var.assign(tf.zeros_like(var))
    loss_after_zero = agent.loss(experience).loss
    self.assertEqual(loss_after_zero, agent.loss(experience).loss)
    self.assertNotEqual(loss_after_zero, loss_after_train)
    # Restore
    checkpointer._checkpoint.restore(checkpointer._manager.latest_checkpoint)
    loss_after_restore = agent.loss(experience).loss
    self.assertNotEqual(loss_after_restore, loss_after_zero)
    self.assertEqual(loss_after_restore, loss_after_train)

  def testAssignAndRestore(self):
    cloning_net = get_dummy_net(self._action_spec)
    self.verifyVariableAssignAndRestore(
        self._observation_spec, self._action_spec, cloning_net)

  # TODO(kbanoop): Add a test where the target network has different values.
  def testLoss(self):
    cloning_net = get_dummy_net(self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001))

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    actions = tf.constant([0, 1], dtype=tf.int32)
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    experience = trajectory.first(
        observation=observations,
        action=actions,
        policy_info=(),
        reward=rewards,
        discount=discounts)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    expected_loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=cloning_net(observations)[0]))

    loss_info = agent.train(experience)
    total_loss = self.evaluate(loss_info.loss)

    self.assertAllClose(total_loss, expected_loss)

    test_util.test_loss_and_train_output(
        test=self,
        expect_equal_loss_values=True,
        agent=agent,
        experience=experience)

  @parameterized.named_parameters(
      ('TrainOnMultipleStepsDist', False, True),
      ('TrainOnMultipleStepsLogits', False, False),
      ('TrainOnSingleStepsDist', True, True),
      ('TrainOnSingleStepsLogits', True, False),
  )
  def testTrainWithNN(self, is_convert, is_distribution_network):
    # Hard code a trajectory shaped (time=6, batch=1, ...).
    traj, time_step_spec, action_spec = create_arbitrary_trajectory()

    if is_convert:
      # Convert to single step trajectory of shapes (batch=6, 1, ...).
      traj = tf.nest.map_structure(common.transpose_batch_time, traj)

    if is_distribution_network:
      cloning_net = sequential.Sequential([
          expand_dims_layer.ExpandDims(-1),
          tf.keras.layers.Dense(action_spec.maximum - action_spec.minimum + 1),
          tf.keras.layers.Lambda(
              lambda t: tfp.distributions.Categorical(logits=t)),
      ])
    else:
      cloning_net = q_network.QNetwork(time_step_spec.observation, action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        time_step_spec,
        action_spec,
        cloning_network=cloning_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
        num_outer_dims=2)
    # Disable clipping to make sure we can see the difference in behavior
    agent.policy._clip = False
    # TODO(b/123883319)
    if tf.executing_eagerly():
      train_and_loss = lambda: agent.train(traj)
    else:
      train_and_loss = agent.train(traj)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    initial_loss = self.evaluate(train_and_loss).loss
    for _ in range(TRAIN_ITERATIONS - 1):
      loss = self.evaluate(train_and_loss).loss

    # We don't necessarily converge to the same actions as in trajectory after
    # 10 steps of an untuned optimizer, but the loss should go down.
    self.assertGreater(initial_loss, loss)

  def testTrainWithSingleOuterDimension(self):
    # Hard code a trajectory shaped (time=6, batch=1, ...).
    traj, time_step_spec, action_spec = create_arbitrary_trajectory()
    # Remove the batch dimension so there is only one outer dimension.
    traj = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=1), traj)

    cloning_net = q_network.QNetwork(time_step_spec.observation, action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        time_step_spec,
        action_spec,
        cloning_network=cloning_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01))
    # Disable clipping to make sure we can see the difference in behavior
    agent.policy._clip = False
    # Remove policy_info, as BehavioralCloningAgent expects none.
    traj = traj.replace(policy_info=())
    # TODO(b/123883319)
    if tf.executing_eagerly():
      train_and_loss = lambda: agent.train(traj)
    else:
      train_and_loss = agent.train(traj)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(TRAIN_ITERATIONS):
      self.evaluate(train_and_loss)
    # Note that we skip the TrajectoryReplay since it requires a time dimension.

  def testTrainWithRNN(self):
    # Hard code a trajectory shaped (time=6, batch=1, ...).
    traj, time_step_spec, action_spec = create_arbitrary_trajectory()
    cloning_net = q_rnn_network.QRnnNetwork(
        time_step_spec.observation, action_spec, lstm_size=(40,))
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        time_step_spec,
        action_spec,
        cloning_network=cloning_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01),
        num_outer_dims=2)
    # Disable clipping to make sure we can see the difference in behavior
    agent.policy._clip = False
    # Remove policy_info, as BehavioralCloningAgent expects none.
    traj = traj.replace(policy_info=())
    # TODO(b/123883319)
    if tf.executing_eagerly():
      train_and_loss = lambda: agent.train(traj)
    else:
      train_and_loss = agent.train(traj)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    initial_loss = self.evaluate(train_and_loss).loss
    for _ in range(TRAIN_ITERATIONS - 1):
      loss = self.evaluate(train_and_loss).loss

    # We don't necessarily converge to the same actions as in trajectory after
    # 10 steps of an untuned optimizer, but the loss should go down.
    self.assertGreater(initial_loss, loss)

  def testPolicy(self):
    cloning_net = get_dummy_net(self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual(
        [2] + self._action_spec.shape.as_list(),
        action_step.action.shape,
    )
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(all(actions_ <= self._action_spec.maximum))
    self.assertTrue(all(actions_ >= self._action_spec.minimum))

  def testInitializeRestoreAgent(self):
    cloning_net = get_dummy_net(self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    checkpoint = tf.train.Checkpoint(agent=agent)

    latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)

    with self.cached_session() as sess:
      checkpoint_load_status.initialize_or_restore(sess)
      self.assertAllEqual(sess.run(action_step.action), [0, 0])


if __name__ == '__main__':
  test_utils.main()
