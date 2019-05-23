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

import numpy as np
import tensorflow as tf

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.environments import trajectory_replay
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# Number of times to train in test loops.
TRAIN_ITERATIONS = 150


class DummyNet(network.Network):

  def __init__(self, unused_observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(
        unused_observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self._layers.append(
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant([[2, 1],
                                                                   [1, 1]]),
            bias_initializer=tf.compat.v1.initializers.constant([[1], [1]]),
            dtype=tf.float32))

  def call(self, inputs, unused_step_type=None, network_state=()):
    inputs = tf.cast(inputs[0], tf.float32)
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs, network_state


class BehavioralCloningAgentTest(tf.test.TestCase):

  def setUp(self):
    super(BehavioralCloningAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = [tensor_spec.TensorSpec([2], tf.float32)]
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = [tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)]
    self._observation_spec = self._time_step_spec.observation

  def testCreateAgent(self):
    cloning_net = DummyNet(self._observation_spec, self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)
    self.assertIsNotNone(agent.policy)

  def testCreateAgentNestSizeChecks(self):
    action_spec = [
        tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1),
        tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)
    ]

    cloning_net = DummyNet(self._observation_spec, action_spec)
    with self.assertRaisesRegexp(NotImplementedError, '.*Multi-arity.*'):
      behavioral_cloning_agent.BehavioralCloningAgent(
          self._time_step_spec,
          action_spec,
          cloning_network=cloning_net,
          optimizer=None)

  def testCreateAgentDimChecks(self):
    action_spec = [tensor_spec.BoundedTensorSpec([1, 2], tf.int32, 0, 1)]
    cloning_net = DummyNet(self._observation_spec, action_spec)
    with self.assertRaisesRegexp(NotImplementedError, '.*one dimensional.*'):
      behavioral_cloning_agent.BehavioralCloningAgent(
          self._time_step_spec, action_spec,
          cloning_network=cloning_net,
          optimizer=None)

  # TODO(kbanoop): Add a test where the target network has different values.
  def testLoss(self):
    cloning_net = DummyNet(self._observation_spec, self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    actions = [tf.constant([0, 1], dtype=tf.int32)]
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    experience = trajectory.first(
        observation=observations,
        action=actions,
        policy_info=(),
        reward=rewards,
        discount=discounts)
    loss_info = agent._loss(experience)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    total_loss, _ = self.evaluate(loss_info)

    expected_loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=cloning_net(observations)[0], labels=actions[0]))

    self.assertAllClose(total_loss, expected_loss)

  def testTrain(self):
    # Emits trajectories shaped (batch=1, time=6, ...)
    traj, time_step_spec, action_spec = (
        driver_test_utils.make_random_trajectory())
    # Convert to shapes (batch=6, 1, ...) so this works with a non-RNN model.
    traj = tf.nest.map_structure(common.transpose_batch_time, traj)
    cloning_net = q_network.QNetwork(
        time_step_spec.observation, action_spec)
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
    replay = trajectory_replay.TrajectoryReplay(agent.policy)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    initial_actions = self.evaluate(replay.run(traj)[0])
    for _ in range(TRAIN_ITERATIONS):
      self.evaluate(train_and_loss)
    post_training_actions = self.evaluate(replay.run(traj)[0])
    # We don't necessarily converge to the same actions as in trajectory after
    # 10 steps of an untuned optimizer, but the policy does change.
    self.assertFalse(np.all(initial_actions == post_training_actions))

  def testTrainWithRNN(self):
    # Emits trajectories shaped (batch=1, time=6, ...)
    traj, time_step_spec, action_spec = (
        driver_test_utils.make_random_trajectory())
    cloning_net = q_rnn_network.QRnnNetwork(
        time_step_spec.observation, action_spec)
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
    replay = trajectory_replay.TrajectoryReplay(agent.policy)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    initial_actions = self.evaluate(replay.run(traj)[0])

    for _ in range(TRAIN_ITERATIONS):
      self.evaluate(train_and_loss)
    post_training_actions = self.evaluate(replay.run(traj)[0])
    # We don't necessarily converge to the same actions as in trajectory after
    # 10 steps of an untuned optimizer, but the policy does change.
    self.assertFalse(np.all(initial_actions == post_training_actions))

  def testPolicy(self):
    cloning_net = DummyNet(self._observation_spec, self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)
    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual(
        [2] + self._action_spec[0].shape.as_list(),
        action_step.action[0].shape,
    )
    self.evaluate(tf.compat.v1.initialize_all_variables())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(all(actions_[0] <= self._action_spec[0].maximum))
    self.assertTrue(all(actions_[0] >= self._action_spec[0].minimum))

  def testInitializeRestoreAgent(self):
    cloning_net = DummyNet(self._observation_spec, self._action_spec)
    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        cloning_network=cloning_net,
        optimizer=None)
    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.initialize_all_variables())

    checkpoint = tf.train.Checkpoint(agent=agent)

    latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)

    with self.cached_session() as sess:
      checkpoint_load_status.initialize_or_restore(sess)
      self.assertAllEqual(sess.run(action_step.action), [[0, 0]])


if __name__ == '__main__':
  tf.test.main()
