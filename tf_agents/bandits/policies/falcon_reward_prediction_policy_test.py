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

"""Test for falcon_reward_prediction_policy."""

from typing import List, Optional, Set

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import falcon_reward_prediction_policy
from tf_agents.networks import network
from tf_agents.policies import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal


class DummyNet(network.Network):

  def __init__(self, observation_spec, num_actions=3):
    super(DummyNet, self).__init__(observation_spec, (), 'DummyNet')

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer(
                [[1, 1.5, 2], [1, 1.5, 4]]
            ),
            bias_initializer=tf.constant_initializer([[1], [1], [-10]]),
        )
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class GetNumberOfTrainableElementsTest(test_utils.TestCase):

  def testNumberOfTrainableElements(self):
    dummy_net = DummyNet(tensor_spec.TensorSpec([2], tf.float32))
    dummy_net.create_variables()
    self.assertEqual(
        falcon_reward_prediction_policy.get_number_of_trainable_elements(
            dummy_net
        ),
        9,
    )


ALLOWED_VALUES_CASES = (
    {'testcase_name': 'NoMask', 'mask': None},
    {'testcase_name': 'Action_0_Allowed', 'mask': [1, 0, 0]},
    {'testcase_name': 'Action_1_Allowed', 'mask': [0, 1, 0]},
    {'testcase_name': 'Action_2_Allowed', 'mask': [0, 0, 1]},
    {'testcase_name': 'Actions_0_And_1_Allowed', 'mask': [1, 1, 0]},
    {'testcase_name': 'Actions_0_And_2_Allowed', 'mask': [1, 0, 1]},
    {'testcase_name': 'Actions_1_And_2_Allowed', 'mask': [0, 1, 1]},
    {'testcase_name': 'All_Actions_Allowed', 'mask': [1, 1, 1]},
)


@test_util.run_all_in_graph_and_eager_modes
class FalconRewardPredictionPolicyTest(
    test_utils.TestCase, parameterized.TestCase
):

  def _check_uniform_actions(
      self, actions: np.ndarray, allowed_actions: Set[int]
  ) -> None:
    self.assertAllInSet(actions, allowed_actions)
    # Set tolerance in the chosen count to be 4 std.
    num_allow_actions = len(allowed_actions)
    prob = 1.0 / num_allow_actions
    batch_size = actions.shape[0]
    tol = 4.0 * np.sqrt(batch_size * prob * (1.0 - prob))
    expected_count = int(batch_size * prob)
    for action in list(allowed_actions):
      action_chosen_count = np.sum(actions == action)
      self.assertNear(
          action_chosen_count,
          expected_count,
          tol,
          msg=(
              f'action: {action} is expected to be chosen between'
              f' {expected_count - tol} and {expected_count + tol} times, but'
              f' was actually chosen {action_chosen_count} times.'
          ),
      )

  def setUp(self):
    super(FalconRewardPredictionPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    self._time_step_with_mask_spec = ts.time_step_spec(
        (self._obs_spec, tensor_spec.TensorSpec([3], tf.int32))
    )

  def testBanditPolicyType(self):
    policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        emit_policy_info=(utils.InfoFields.BANDIT_POLICY_TYPE,),
    )
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    self.assertAllEqual(
        p_info.bandit_policy_type,
        [[utils.BanditPolicyType.FALCON], [utils.BanditPolicyType.FALCON]],
    )

  def _create_policy(
      self,
      use_mask: bool,
      num_samples_list: List[tf.compat.v2.Variable],
      exploitation_coefficient: Optional[float] = 1.0,
      max_exploration_probability_hint: Optional[float] = None,
  ) -> falcon_reward_prediction_policy.FalconRewardPredictionPolicy:
    if use_mask:

      def split_fn(obs):
        return obs[0], obs[1]

      policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
          time_step_spec=self._time_step_with_mask_spec,
          action_spec=self._action_spec,
          reward_network=DummyNet(self._obs_spec),
          exploitation_coefficient=exploitation_coefficient,
          max_exploration_probability_hint=max_exploration_probability_hint,
          num_samples_list=num_samples_list,
          emit_policy_info=(
              utils.InfoFields.LOG_PROBABILITY,
              utils.InfoFields.PREDICTED_REWARDS_MEAN,
          ),
          observation_and_action_constraint_splitter=split_fn,
      )
    else:
      policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
          time_step_spec=self._time_step_spec,
          action_spec=self._action_spec,
          reward_network=DummyNet(self._obs_spec),
          exploitation_coefficient=exploitation_coefficient,
          max_exploration_probability_hint=max_exploration_probability_hint,
          num_samples_list=num_samples_list,
          emit_policy_info=(
              utils.InfoFields.LOG_PROBABILITY,
              utils.InfoFields.PREDICTED_REWARDS_MEAN,
          ),
      )
    return policy

  @parameterized.named_parameters(*ALLOWED_VALUES_CASES)
  def testZeroExploitationCoefficient(self, mask):
    # With a zero exploitation coefficient, the sampling probability will be
    # uniform.
    policy = self._create_policy(
        use_mask=(mask is not None),
        exploitation_coefficient=0.0,
        num_samples_list=[
            tf.compat.v2.Variable(2, dtype=tf.int32, name='num_samples_0'),
            tf.compat.v2.Variable(4, dtype=tf.int32, name='num_samples_1'),
            tf.compat.v2.Variable(1, dtype=tf.int32, name='num_samples_2'),
        ],
    )
    batch_size = 3000
    if mask is None:
      observations = tf.constant([[1, 2]] * batch_size, dtype=tf.float32)
    else:
      observations = (
          tf.constant([[1, 2]] * batch_size, dtype=tf.float32),
          tf.constant([mask] * batch_size, dtype=tf.float32),
      )
    time_step = ts.restart(observations, batch_size=batch_size)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are uniform and the
    # empirical distribution of the chosen arms is uniform.
    actions = self.evaluate(action_step.action)
    if mask is None:
      self.assertAllClose(
          p_info.log_probability, tf.math.log([1.0 / 3] * batch_size)
      )
      self._check_uniform_actions(actions=actions, allowed_actions=[0, 1, 2])
    else:
      self.assertAllClose(
          p_info.log_probability, tf.math.log([1.0 / np.sum(mask)] * batch_size)
      )
      self._check_uniform_actions(
          actions=actions, allowed_actions=np.nonzero(mask)[0]
      )

  def testLargeExploitationCoefficient(self):
    # With a very large exploitation coefficient and a positive number of
    # samples, the sampled actions will be greedy.
    policy = self._create_policy(
        use_mask=False,
        exploitation_coefficient=10.0**20,
        num_samples_list=[
            tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_0'),
            tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_1'),
            tf.compat.v2.Variable(10, dtype=tf.int32, name='num_samples_2'),
        ],
    )
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are greedy.
    self.assertAllClose(p_info.log_probability, [0.0, 0.0], atol=1e-4)
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  @parameterized.product(
      ALLOWED_VALUES_CASES, set_max_exploration_probability_hint=(True, False)
  )
  def testZeroNumSamples(
      self, testcase_name, mask, set_max_exploration_probability_hint
  ):
    del testcase_name
    # With number of samples being 0, the sampling probabiity will be uniform.
    if set_max_exploration_probability_hint:
      policy = self._create_policy(
          use_mask=(mask is not None),
          max_exploration_probability_hint=0.01,
          num_samples_list=[
              tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_0'),
              tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_1'),
              tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_2'),
          ],
      )
    else:
      policy = self._create_policy(
          use_mask=(mask is not None),
          exploitation_coefficient=100.0,
          num_samples_list=[
              tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_0'),
              tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_1'),
              tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_2'),
          ],
      )
    batch_size = 3000
    if mask is None:
      observations = tf.constant([[1, 2]] * batch_size, dtype=tf.float32)
    else:
      observations = (
          tf.constant([[1, 2]] * batch_size, dtype=tf.float32),
          tf.constant([mask] * batch_size, dtype=tf.float32),
      )
    time_step = ts.restart(observations, batch_size=batch_size)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are uniform and the
    # empirical distribution of the chosen arms is uniform.
    actions = self.evaluate(action_step.action)
    if mask is None:
      self.assertAllClose(
          p_info.log_probability, tf.math.log([1.0 / 3] * batch_size)
      )
      self._check_uniform_actions(actions=actions, allowed_actions=[0, 1, 2])
    else:
      self.assertAllClose(
          p_info.log_probability, tf.math.log([1.0 / np.sum(mask)] * batch_size)
      )
      self._check_uniform_actions(
          actions=actions, allowed_actions=np.nonzero(mask)[0]
      )

  @parameterized.parameters(True, False)
  def testLargeNumSamples(self, set_max_exploration_probability_hint):
    # With very large numbers of samples and a positive learning rate, the
    # sampled actions will be greedy.
    if set_max_exploration_probability_hint:
      policy = self._create_policy(
          use_mask=False,
          max_exploration_probability_hint=0.05,
          num_samples_list=[
              tf.compat.v2.Variable(
                  int(1e10), dtype=tf.int32, name='num_samples_0'
              ),
              tf.compat.v2.Variable(
                  int(1e10), dtype=tf.int32, name='num_samples_1'
              ),
              tf.compat.v2.Variable(
                  int(1e10), dtype=tf.int32, name='num_samples_2'
              ),
          ],
      )
    else:
      policy = self._create_policy(
          use_mask=False,
          exploitation_coefficient=0.1,
          num_samples_list=[
              tf.compat.v2.Variable(
                  int(1e10), dtype=tf.int32, name='num_samples_0'
              ),
              tf.compat.v2.Variable(
                  int(1e10), dtype=tf.int32, name='num_samples_1'
              ),
              tf.compat.v2.Variable(
                  int(1e10), dtype=tf.int32, name='num_samples_2'
              ),
          ],
      )
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are greedy.
    self.assertAllClose(p_info.log_probability, [0.0, 0.0], atol=1e-3)
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  @parameterized.named_parameters(*ALLOWED_VALUES_CASES)
  def testMaxExplorationProbabilityHint(self, mask):
    batch_size = 1000
    # Constructs a batch of observations such that action 1 is always the greedy
    # choice, but its gap in predicted rewards from the second-best action may
    # vary.
    features = tf.stack(
        values=[
            tf.constant(1e-2, shape=[batch_size], dtype=tf.float32),
            tf.random.uniform(shape=[batch_size], maxval=1.0, dtype=tf.float32),
        ],
        axis=1,
    )
    if mask is None:
      observations = features
    else:
      observations = (
          features,
          tf.constant([mask] * batch_size, dtype=tf.float32),
      )

    time_step = ts.restart(observations, batch_size=batch_size)

    # Creates a policy with a low sample count, a small
    # `exploitation_coefficient` and unset `max_exploration_probability_hint`.
    # The best allowed action is expected to be chosen the most, but not at a
    # very high probability.
    policy = self._create_policy(
        use_mask=(mask is not None),
        exploitation_coefficient=0.1,
        num_samples_list=[
            tf.compat.v2.Variable(int(1), dtype=tf.int32, name='num_samples_0'),
            tf.compat.v2.Variable(int(1), dtype=tf.int32, name='num_samples_1'),
            tf.compat.v2.Variable(int(1), dtype=tf.int32, name='num_samples_2'),
        ],
    )
    action_step = policy.action(time_step, seed=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    if mask is None or np.sum(mask) == 3:
      # All actions are allowed, action 1 is the best.
      self.assertLess(np.sum(actions == 1), 0.5 * batch_size)
      self.assertGreater(np.sum(actions == 1), np.sum(actions == 0))
      self.assertGreater(np.sum(actions == 1), np.sum(actions == 2))
    elif np.sum(mask) == 1:
      # A single action is allowed.
      only_allowed_action = np.argwhere(mask)[0]
      self.assertEqual(np.sum(actions == only_allowed_action), batch_size)
    else:
      # Two actions are allowed.
      self.assertEqual(np.sum(mask), 2)
      action_counts = [np.sum(actions == action) for action in range(3)]
      most_chosen_action = np.argmax(action_counts)
      # The worst action 2 is expected to never be the most chosen.
      self.assertNotEqual(most_chosen_action, 2)
      self.assertLessEqual(action_counts[most_chosen_action], 0.7 * batch_size)

    # Creates a policy with the same sample count and
    # `exploitation_coefficient`, but sets `max_exploration_probability_hint` to
    # 5%. The best allowed action is expected to be chosen 95% of the time.
    limited_exploration_policy = self._create_policy(
        use_mask=(mask is not None),
        exploitation_coefficient=0.1,
        max_exploration_probability_hint=0.05,
        num_samples_list=[
            tf.compat.v2.Variable(int(1), dtype=tf.int32, name='num_samples_0'),
            tf.compat.v2.Variable(int(1), dtype=tf.int32, name='num_samples_1'),
            tf.compat.v2.Variable(int(1), dtype=tf.int32, name='num_samples_2'),
        ],
    )
    action_step = limited_exploration_policy.action(time_step, seed=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    action_counts = [np.sum(actions == action) for action in range(3)]
    most_chosen_action = np.argmax(action_counts)
    # Sets the threshold to be a bit lower than 95% for test robustness against
    # random sampling.
    self.assertGreater(action_counts[most_chosen_action], 0.9 * batch_size)
    if mask is None or np.sum(mask) == 3:
      # All actions are allowed, the greedy action (1) is expected to be chosen
      # the most.
      self.assertEqual(most_chosen_action, 1)
    elif np.sum(mask) == 1:
      # A single action is allowed, and is expected to always be chosen.
      only_allowed_action = int(np.argwhere(mask)[0])
      self.assertEqual(most_chosen_action, only_allowed_action)
      self.assertEqual(action_counts[only_allowed_action], batch_size)
    else:
      # Two actions are allowed.
      self.assertEqual(np.sum(mask), 2)
      # The greedy action is action 1 if allowed, or action 0 otherwise.
      expected_greedy_action = 1 if mask[1] > 0 else 0
      self.assertEqual(most_chosen_action, expected_greedy_action)

  def testMaxExplorationProbabilityHintAmbiguousRewards(self):
    batch_size = 1000
    # Constructs a batch of observations such that action 1 is always the greedy
    # choice, action 0 is almost as good, and action 2 is much worse in
    # predicted rewards.
    observations = tf.stack(
        values=[
            tf.constant(1e-5, shape=[batch_size], dtype=tf.float32),
            tf.random.uniform(
                shape=[batch_size], maxval=1e-5, dtype=tf.float32
            ),
        ],
        axis=1,
    )
    time_step = ts.restart(observations, batch_size=batch_size)

    # Creates a policy with `max_exploration_probability_hint`to 5%, which is
    # expected to apply only to action 2 because it has much worse predicted
    # rewards than actions 1 (greedy) and 0 (runner-up). Action 1 is expected to
    # be chosen the most often, while action 0 is expected to be chosen with
    # probability close to 1.0 / num_actions.
    limited_exploration_policy = self._create_policy(
        use_mask=False,
        exploitation_coefficient=10.0,
        max_exploration_probability_hint=0.05,
        num_samples_list=[
            tf.compat.v2.Variable(
                int(500000), dtype=tf.int32, name='num_samples_0'
            ),
            tf.compat.v2.Variable(
                int(500000), dtype=tf.int32, name='num_samples_1'
            ),
            tf.compat.v2.Variable(
                int(500000), dtype=tf.int32, name='num_samples_2'
            ),
        ],
    )
    action_step = limited_exploration_policy.action(time_step, seed=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    p_info = self.evaluate(action_step.info)
    print(p_info.predicted_rewards_mean)
    self.assertGreater(np.sum(actions == 0), 0.3 * batch_size)
    self.assertGreater(np.sum(actions == 1), np.sum(actions == 0))
    # Sets the upperbound to be a bit larger than expected for test robustness
    # against random sampling.
    self.assertLess(np.sum(actions == 2), 0.1 * batch_size)


if __name__ == '__main__':
  tf.test.main()
