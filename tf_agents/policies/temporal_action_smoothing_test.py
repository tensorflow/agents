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

"""Tests for tf_agents.policies.temporal_action_smoothing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import temporal_action_smoothing
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class StateIncrementPolicy(tf_policy.TFPolicy):

  def __init__(self, time_step_spec, action_spec):
    super(StateIncrementPolicy, self).__init__(
        time_step_spec,
        action_spec,
        policy_state_spec=action_spec,
    )

  def _action(self, time_step, policy_state, seed):
    actions = tf.nest.map_structure(lambda t: t + 1, policy_state)
    return policy_step.PolicyStep(actions, actions, ())

  def _distribution(self):
    return policy_step.PolicyStep(())


class TemporalActionSmoothingTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(TemporalActionSmoothingTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 0, 10)

  @property
  def _time_step(self):
    return ts.transition(tf.constant([[1, 2]], dtype=tf.float32),
                         reward=tf.constant([1.]))

  def testStateIncrementPolicy(self):
    policy = StateIncrementPolicy(self._time_step_spec, self._action_spec)
    policy_state = policy.get_initial_state(1)
    step = policy.action(self._time_step, policy_state)
    self.assertEqual(1, self.evaluate(step.action))
    step = policy.action(self._time_step, step.state)
    self.assertEqual(2, self.evaluate(step.action))

  @parameterized.named_parameters(
      ('0p0', 0.0, [1., 2., 3., 4., 5.]),
      ('0p5', 0.5, [0.5, 1.25, 2.125, 3.0625, 4.03125]),
      ('1p0', 1.0, [0., 0., 0., 0., 0.]),
  )
  def testSmoothedActions(self, smoothing_coefficient, expected_actions):
    # Set up the smoothing policy.
    policy = StateIncrementPolicy(self._time_step_spec, self._action_spec)
    smoothed_policy = temporal_action_smoothing.TemporalActionSmoothing(
        policy, smoothing_coefficient)

    # Create actions sampled in time order.
    policy_state = smoothed_policy.get_initial_state(batch_size=1)
    smoothed_actions = []
    for _ in range(5):
      action, policy_state, unused_policy_info = smoothed_policy.action(
          self._time_step, policy_state=policy_state)
      smoothed_actions.append(action)

    # Make sure smoothed actions are as expected.
    smoothed_actions_ = self.evaluate(smoothed_actions)
    self.assertAllClose(np.squeeze(smoothed_actions_), expected_actions)

  def testDistributionRaisesError(self):
    # Set up the smoothing policy.
    policy = StateIncrementPolicy(self._time_step_spec, self._action_spec)
    smoothed_policy = temporal_action_smoothing.TemporalActionSmoothing(
        policy, smoothing_coefficient=0.5)

    # Create actions sampled in time order.
    policy_state = smoothed_policy.get_initial_state(batch_size=1)
    with self.assertRaises(NotImplementedError):
      smoothed_policy.distribution(self._time_step, policy_state)


if __name__ == '__main__':
  tf.test.main()
