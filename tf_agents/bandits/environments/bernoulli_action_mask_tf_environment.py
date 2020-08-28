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

"""Environment wrapper that adds action masks to a bandit environment.

This environment wrapper takes a `BanditTFEnvironment` as input, and generates
a new environment where the observations are joined with boolean action
masks. These masks describe which actions are allowed in a given time step. If a
disallowed action is chosen in a time step, the environment will raise an
error. The masks are drawn independently from Bernoulli-distributed random
variables with parameter `action_probability`.

The observations from the original environment and the mask are joined by the
given `join_fn` function, and the result of the join function will be the
observation in the new environment.

Usage:

 '''
 env = MyFavoriteBanditEnvironment(...)
 def join_fn(context, mask):
   return (context, mask)
 masked_env = BernoulliActionMaskTFEnvironment(env, join_fn, 0.5)
 '''
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Callable

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.environments import bandit_tf_environment
from tf_agents.bandits.policies import policy_utilities
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common

tfd = tfp.distributions


@common.function
def _maybe_add_one_action(mask):
  """For time steps where the mask is all zeros, adds one action randomly."""
  batch_size = tf.shape(mask)[0]
  num_actions = tf.shape(mask)[1]
  extra_actions = tf.one_hot(
      tf.random.uniform([batch_size], 0, num_actions, dtype=tf.int32),
      depth=num_actions,
      dtype=tf.int32)
  cond = tf.cast(tf.equal(tf.reduce_max(mask, axis=1), 1), tf.bool)
  return tf.compat.v1.where(cond, mask, extra_actions)


@gin.configurable
class BernoulliActionMaskTFEnvironment(bandit_tf_environment.BanditTFEnvironment
                                      ):
  """An environment wrapper that adds action masks to observations."""

  def __init__(self,
               original_environment: bandit_tf_environment.BanditTFEnvironment,
               action_constraint_join_fn: Callable[
                   [types.TensorSpec, types.TensorSpec], types.TensorSpec],
               action_probability: float):
    """Initializes a `BernoulliActionMaskTFEnvironment`.

    Args:
      original_environment: Instance of `BanditTFEnvironment`. This environment
        will be wrapped.
      action_constraint_join_fn: A function that joins the osbervation from the
        original environment with the generated masks.
      action_probability: The probability that any action in the action space is
        allower by the generated mask.
    """
    self._original_environment = original_environment
    assert isinstance(
        original_environment, bandit_tf_environment.BanditTFEnvironment
    ), 'The wrapped environment needs to be a `BanditTFEnvironment`.'
    self._action_constraint_join_fn = action_constraint_join_fn
    self._action_probability = action_probability
    self._batch_size = self._original_environment.batch_size
    action_spec = self._original_environment.action_spec()
    observation_spec_without_mask = (
        self._original_environment.time_step_spec().observation)
    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)

    mask_spec = tf.TensorSpec([self._num_actions], dtype=tf.int32)
    joined_observation_spec = self._action_constraint_join_fn(
        observation_spec_without_mask, mask_spec)
    time_step_spec = ts.time_step_spec(joined_observation_spec)

    self._current_mask = tf.compat.v2.Variable(
        tf.ones([self.batch_size, self._num_actions], dtype=tf.int32))

    super(BernoulliActionMaskTFEnvironment, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        batch_size=self._batch_size)

  @property
  def original_environment(self):
    return self._original_environment

  @common.function
  def _check_action_with_mask(self, action):
    is_allowed = tf.gather(
        self._current_mask, tf.expand_dims(action, axis=1), batch_dims=1)
    tf.assert_equal(is_allowed, 1, message='Action not in allowed action set.')

  @common.function
  def _apply_action(self, action):
    self._check_action_with_mask(action)
    # pylint: disable=protected-access
    reward = self.original_environment._apply_action(action)
    return reward

  @common.function
  def _observe(self):
    # pylint: disable=protected-access
    original_observation = self._original_environment._observe()
    mask = tfd.Bernoulli(self._action_probability).sample(
        sample_shape=[self._batch_size, self._num_actions])
    mask = _maybe_add_one_action(mask)
    tf.compat.v1.assign(self._current_mask, mask)

    return self._action_constraint_join_fn(original_observation, mask)
