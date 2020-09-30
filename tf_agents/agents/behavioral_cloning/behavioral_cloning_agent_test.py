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

"""Tests for agents.behavioral_cloning.behavioral_cloning_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
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
  num_actions = flat_action_spec.maximum - flat_action_spec.minimum + 1

  return sequential.Sequential([
      tf.keras.layers.Dense(
          num_actions,
          kernel_initializer=tf.compat.v1.initializers.constant([[2, 1], [1, 1]
                                                                ]),
          bias_initializer=tf.compat.v1.initializers.constant([[1], [1]]),
          dtype=tf.float32)
  ], input_spec=observation_spec)


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
      ('SingleNonScalarFloatAction', [
          tensor_spec.BoundedTensorSpec([3, 2], tf.float32, 0, 1)],
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

  @parameterized.named_parameters(('TrainOnMultipleSteps', False),
                                  ('TrainOnSingleStep', True))
  def testTrainWithNN(self, is_convert):
    # Hard code a trajectory shaped (time=6, batch=1, ...).
    traj, time_step_spec, action_spec = create_arbitrary_trajectory()

    if is_convert:
      # Convert to single step trajectory of shapes (batch=6, 1, ...).
      traj = tf.nest.map_structure(common.transpose_batch_time, traj)
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
