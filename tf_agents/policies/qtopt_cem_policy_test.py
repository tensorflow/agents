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

"""Tests for tf_agents.policies.cem_policy.

Idea is to create Q-functions that have known maximum at certain action values
and verify that the CEM actually returns us an action close to the expected
value.

If anything basic is changed in the CEM model, it is useful to visually inspect
the printed values in the tests and verify that the CEM iterations aer behaving
well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from absl.testing import parameterized

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.policies import qtopt_cem_policy
from tf_agents.policies.samplers import cem_actions_sampler_continuous
from tf_agents.policies.samplers import cem_actions_sampler_continuous_and_one_hot
from tf_agents.policies.samplers import cem_actions_sampler_hybrid
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

FLAGS = flags.FLAGS

# How much distance from the anchor point is acceptable on a range from 0 to 1.
# Chosen based on visual inspection and a roughly ~10% heuristic on the original
# range. In many cases, the actual converged value is much closer to the anchor
# point.
DELTA = 0.2

DIST_L1 = 'distance_l1'
DIST_BIMODAL = 'distance_bimodal_gaussian'
LOCAL_MAX_BIMODAL = 0.7
VAR_ANCHOR_PT = 0.5
VAR_LOCAL_MAX = 4.

_MEAN = [0., 0., 0.]
_VAR = [0.09, 0.03, 0.05]
_BATCH = 2
_ACTION_SIZE = 3
_ACTION_SIZE_DISCRETE = 2


class DummyNet(network.Network):

  def __init__(self, input_tensor_spec, sampler_type=None,
               anchor_point=0.0, dist=DIST_BIMODAL,
               categorical_action_returns=None):
    """Defines a DummyNet class as a simple continuous dist function.

    It has a clear maximum at the specified anchor_point. By default, all
    action dimensions for all batches must be near the anchor point.

    Args:
      input_tensor_spec: Input Tensor Spec.
      sampler_type: One of 'continuous', 'continuous_and_one_hot' and 'hybrid'.
      anchor_point: The point around which we want the Q-function to have a max.
        Independently applied to each action dimension for each sample in the
        batch.
      dist: If DIST_L1, then negative of L1 distance from anchor point is used
        for q_value. DIST_BIMODAL, a bimodal Gaussian function is used:
        one larger mode centered around the anchor point and one smaller mode
        centered around LOCAL_MAX_BIMODAL. In both cases, the q_func should have
        a maxima at the specified anchor point.
      categorical_action_returns: If not None, tells the rewards for each
        categorical action. So [10., 0.] would mean that first categorical
        action is very important. [0., 10.] will indicate the second one is
        very important. So the CEM is expected to output the respective one
        as the best possible action instead of something close to the anchor
        point.
    """
    self._sampler_type = sampler_type
    self._anchor_point = anchor_point
    self._dist = dist
    self._categorical_action_returns = categorical_action_returns

    _, action_spec = input_tensor_spec
    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) == 2:
      if flat_action_spec[0].dtype.is_floating:
        self.idx_continuous = 0
      else:
        self.idx_continuous = 1
    else:
      self.idx_continuous = 0

    super(DummyNet, self).__init__(
        input_tensor_spec, (), 'DummyNet')

  def call(self, inputs, step_type=(), network_state=()):
    _, action_nested = inputs

    flat_action = tf.nest.flatten(action_nested)

    val = self._get_proximity_to_anchor(
        flat_action[self.idx_continuous], self._anchor_point, self._dist)
    res = tf.reduce_min(val, axis=-1)

    # If there are categorical actions, then make the Q value very high
    # when they have high returns in categorical_action_returns.
    # The expectation is that CEM will find this the best
    # possible action since this Q value dominates the ones coming from
    # proximity to the given anchor point.
    if self._sampler_type == 'continuous_and_one_hot':
      cat_actions_dist_1 = tf.abs(flat_action[1-self.idx_continuous] - 1)
      cat_returns = tf.constant(self._categorical_action_returns)
      overall_cat_actions_vals = -tf.multiply(
          cat_returns, tf.cast(cat_actions_dist_1, tf.float32))
      res += tf.reduce_min(overall_cat_actions_vals, axis=1)
    elif self._sampler_type == 'hybrid':
      val_discrete = self._get_proximity_to_anchor(
          flat_action[1 - self.idx_continuous], self._anchor_point, self._dist)
      res += tf.reduce_min(val_discrete, axis=-1)

    return res, ()

  def _get_proximity_to_anchor(self, action, anchor_point, dist):
    delta = tf.cast(action, tf.float32) - anchor_point
    logging.info('Test q_func \n' 'anchor: %s, dist: %s', anchor_point, dist)
    if dist == DIST_L1:
      val = -tf.abs(delta)
    elif dist == DIST_BIMODAL:
      val = (
          tfp.distributions.Normal(0., VAR_ANCHOR_PT).prob(delta) +
          tfp.distributions.Normal(
              0., VAR_LOCAL_MAX).prob(action - LOCAL_MAX_BIMODAL))
    else:
      raise ValueError('Unknown distance option for Q-function definition.')
    return val


class CEMPolicyTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(CEMPolicyTest, self).setUp()
    seed = 1999
    logging.info('Setting the tf seed to %d', seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        [_ACTION_SIZE], tf.float32, 0.0, 1.0)
    self._action_spec_hybrid = {
        'continuous': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE - _ACTION_SIZE_DISCRETE], tf.float32, 0.0, 1.0),
        'discrete': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_DISCRETE], tf.int32, 0, 1)
    }

  def testBuild(self):
    policy = qtopt_cem_policy.CEMPolicy(
        self._time_step_spec, self._action_spec, sampler=None, init_mean=None,
        init_var=None, q_network=DummyNet((self._obs_spec, self._action_spec)))

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  def test_initial_params(self):
    policy = qtopt_cem_policy.CEMPolicy(
        self._time_step_spec, self._action_spec, sampler=None, init_mean=_MEAN,
        init_var=_VAR, q_network=DummyNet((self._obs_spec, self._action_spec)))

    mean, var = policy._initial_params(_BATCH)

    with self.session(use_gpu=True):
      self.assertAllClose([[0., 0., 0.], [0., 0., 0.]], mean)
    self.assertAllClose([[0.09, 0.03, 0.05], [0.09, 0.03, 0.05]], var)

  @parameterized.named_parameters(
      {
          'sampler_type': 'continuous',
          'testcase_name': 'L1_0',
          'anchor_point': 0.,
          'dist': DIST_L1
      },
      {
          'sampler_type': 'continuous',
          'testcase_name': 'L1_1',
          'anchor_point': 1.,
          'dist': DIST_L1
      },
      {
          'sampler_type': 'continuous',
          'testcase_name': 'L1_0pt5',
          'anchor_point': 0.5,
          'dist': DIST_L1
      },
      {
          'sampler_type': 'continuous',
          'testcase_name': 'bimodal_0',
          'anchor_point': 0.,
          'dist': DIST_BIMODAL
      },
      {
          'sampler_type': 'continuous',
          'testcase_name': 'bimodal_1',
          'anchor_point': 1.,
          'dist': DIST_BIMODAL
      },
      {
          'sampler_type': 'continuous',
          'testcase_name': 'bimodal_0pt5',
          'anchor_point': 0.5,
          'dist': DIST_BIMODAL
      },
      {
          'sampler_type': 'hybrid',
          'testcase_name': 'L1_0_hybrid',
          'anchor_point': 0.,
          'dist': DIST_L1
      },
      {
          'sampler_type': 'continuous_and_one_hot',
          'testcase_name': 'L1_pt5_cat0',
          'anchor_point': 0.5,
          'dist': DIST_L1,
          'categorical_action_returns': [10., 0.]
      },
      {
          'sampler_type': 'continuous_and_one_hot',
          'testcase_name': 'bimodal_pt5_cat1',
          'anchor_point': 0.5,
          'dist': DIST_BIMODAL,
          'categorical_action_returns': [100., 10.]
      },
  )
  def test_actor_func(self,
                      sampler_type=None,
                      anchor_point=0.,
                      dist=DIST_BIMODAL,
                      categorical_action_returns=None,
                      num_iters=5):  # pylint: disable=g-doc-args
    """Helper function to run the tests.

    Creates the right q_func and tests for correctness. See the _create_q_func
    documentation to understand the various arguments.
    """
    logging.info('==== Anchor point = %.2f ====', anchor_point)
    logging.info('dist = %s', dist)

    # we need more samples for test to guarantee convergence.
    num_samples = 64 * 4

    if sampler_type == 'continuous_and_one_hot':
      action_spec = self._action_spec_hybrid
      # Initiate random mean and var.
      np.random.seed(1999)
      samples = (np.random.rand(num_samples,
                                _ACTION_SIZE - _ACTION_SIZE_DISCRETE).astype(
                                    np.float32))  # [N, A-S]
      mean = {
          'continuous': np.mean(samples, axis=0),
          'discrete': np.zeros([_ACTION_SIZE_DISCRETE, _ACTION_SIZE_DISCRETE],
                               dtype=np.float32)
      }
      var = {
          'continuous': np.var(samples, axis=0),
          'discrete': np.zeros([_ACTION_SIZE_DISCRETE, _ACTION_SIZE_DISCRETE],
                               dtype=np.float32)
      }

      sampler = cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
          action_spec=action_spec, sample_clippers=[[], []],
          sub_actions_fields=[['discrete'], ['continuous']])
    elif sampler_type == 'hybrid':
      action_spec = self._action_spec_hybrid
      # Initiate random mean and var.
      np.random.seed(1999)
      samples = (np.random.rand(num_samples,
                                _ACTION_SIZE - _ACTION_SIZE_DISCRETE).astype(
                                    np.float32))  # [N, A-S]
      samples_discrete = (np.random.rand(
          num_samples, _ACTION_SIZE_DISCRETE).astype(np.float32))  # [N, A-S]
      mean = {
          'continuous': np.mean(samples, axis=0),
          'discrete': np.mean(samples_discrete, axis=0),
      }
      var = {
          'continuous': np.var(samples, axis=0),
          'discrete': np.var(samples_discrete, axis=0),
      }

      sampler = cem_actions_sampler_hybrid.GaussianActionsSampler(
          action_spec=action_spec)
    elif sampler_type == 'continuous':
      action_spec = self._action_spec
      # Initiate random mean and var.
      np.random.seed(1999)
      samples = (np.random.rand(num_samples,
                                _ACTION_SIZE).astype(np.float32))  # [N, A]
      mean = np.mean(samples, axis=0).tolist()
      var = np.var(samples, axis=0).tolist()
      sampler = cem_actions_sampler_continuous.GaussianActionsSampler(
          action_spec=action_spec)

    cem_fn = qtopt_cem_policy.CEMPolicy(
        self._time_step_spec,
        action_spec,
        q_network=DummyNet(
            (self._obs_spec, action_spec),
            sampler_type=sampler_type,
            anchor_point=anchor_point,
            dist=dist,
            categorical_action_returns=categorical_action_returns),
        sampler=sampler,
        init_mean=mean,
        init_var=var,
        num_samples=num_samples,
        num_elites=7,
        num_iterations=num_iters)
    out, _, _ = cem_fn.actor_func(
        observation=tf.zeros([_BATCH, 2]), step_type=(), policy_state=())

    self._assert_correctness(
        out,
        sampler_type=sampler_type,
        anchor_point=anchor_point,
        categorical_action_returns=categorical_action_returns)

  def _assert_correctness(self, res, sampler_type, anchor_point=0.,
                          categorical_action_returns=None):
    logging.info('anchor = %.2f', anchor_point)
    logging.info('Output best actions (size [batch, action])\n %s', res)

    if sampler_type == 'continuous_and_one_hot':
      num_categorical_actions = len(categorical_action_returns)
      argmax_action = np.argmax(categorical_action_returns)

      res = self.evaluate(res)

      # First ensure all continuous parts are 0 for whole batch.
      self.assertEqual(res['continuous'].all(), 0.)
      # Ensure the right categorical action is chosen for whole batch.
      expected = [
          1 if i == argmax_action else 0
          for i in range(num_categorical_actions)
      ]
      expected = np.stack([expected for i in range(len(res))])
      self.assertAllEqual(expected, res['discrete'][:, 0, :])
    elif sampler_type == 'hybrid':
      # All actions must be close to the anchor point.
      self.assertAllLessEqual(
          tf.abs(tf.cast(res['discrete'], tf.float32) - anchor_point), DELTA)
      self.assertAllLessEqual(tf.abs(res['continuous'] - anchor_point), DELTA)
    elif sampler_type == 'continuous':
      # All actions must be close to the anchor point.
      self.assertAllLessEqual(tf.abs(res - anchor_point), DELTA)

  @parameterized.parameters((1, 1, 0.0), (4, 1, 0.5), (64, 1, 1.0),
                            (1, 1, 0.66), (4, 1, -0.765),
                            (64, 1, 1.8), (4, 2, 0.5))
  def test_score(self, num_cem_samples, seq_size, anchor_point):
    # Test that the scores are same now as they were with the map_fn in the
    # cem while loop.
    batch_size = 2
    action_size = 8
    num_iters = 3
    obs_spec = tf.TensorSpec([1], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec([action_size], tf.float32, 0.0,
                                                1.0)
    cem_agent = qtopt_cem_policy.CEMPolicy(
        self._time_step_spec,
        self._action_spec,
        q_network=DummyNet((obs_spec, action_spec), anchor_point=anchor_point),
        num_samples=num_cem_samples,
        init_mean=None,
        init_var=None,
        sampler=None,
        num_iterations=num_iters)
    sample_actions = tf.random.uniform(
        [batch_size*seq_size, num_cem_samples, action_size])  # [BxT, N, A]

    # Calculate the scores [without the map_fn]
    if seq_size == 1:
      scores, _ = cem_agent._score(
          observation=tf.zeros([batch_size, seq_size, 1]),
          sample_actions=sample_actions,
          step_type=(),
          policy_state=())
    else:
      scores, _ = cem_agent._score_with_time(
          observation=tf.zeros([batch_size, seq_size, 1]),
          sample_actions=sample_actions,
          step_type=(),
          policy_state=(),
          seq_size=seq_size)

    # Calculate scores with the map_fn
    orig_scores = self._score_with_map_fn(
        observation=tf.zeros([batch_size, 1]),
        sample_actions=tf.transpose(sample_actions, [1, 0, 2]),
        q_network=DummyNet((obs_spec, action_spec), anchor_point=anchor_point))

    self.assertAllEqual(scores.shape, (batch_size*seq_size, num_cem_samples))
    logging.info(scores)
    logging.info(orig_scores)

    # Evaluates scores and orig_scores at the same time so that they share the
    # same random inputs.
    scores_value, orig_scores_value = self.evaluate([scores, orig_scores])
    self.assertAllClose(scores_value, orig_scores_value)

  def _score_with_map_fn(self, observation, sample_actions, q_network):
    """Scores the sample actions using map_fn as part of CEM while loop.

    Args:
      observation: A batch of observation tensors or NamedTuples, whatever the
        q_func will handle. CEM is agnostic to it.
      sample_actions: A [N, B, A] sized tensor, where batch is the batch
        size, N is the sample size for the CEM, a is the size of the action
        space.
      q_network: the q_network to use for calculating scores.

    Returns:
      a tensor of shape [B, N] representing the scores for the actions
    """

    # Get the score of a single sample_action: [B, A]
    def single_sample_score_fn(sample_action):
      """Gets the score of a single sample action for the observation."""
      q_values, _ = q_network(
          (observation, sample_action))  # [B]

      return q_values

    # Main _score function.
    # 1. Observation (including the image) should be reused for each of the
    #    actions.
    # 2. Call the q_func with the action, observation combo.
    # sample_actions  # [N, B, A]
    scores_t = tf.map_fn(
        single_sample_score_fn, sample_actions,
        dtype=tf.float32)  # [N, B]
    # Note that we only want to squeeze the last dimension in case the batch=1.
    # Else, while loop will throw an error since the samples enter the loop as
    # (1, N, 1) and will exit (N,).
    return tf.transpose(scores_t, [1, 0])  # [B, N]


if __name__ == '__main__':
  tf.test.main()
