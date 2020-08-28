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

"""Bandit environment that returns random observations and rewards."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.environments import bandit_tf_environment as bte
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.typing import types

__all__ = ['RandomBanditEnvironment']


def _raise_batch_shape_error(distribution_name, batch_shape):
  raise ValueError('`{distribution_name}` must have batch shape with length 1; '
                   'got {batch_shape}. Consider using '
                   '`tensorflow_probability.distributions.Independent` '
                   'to manipulate batch and event shapes.'.format(
                       distribution_name=distribution_name,
                       batch_shape=batch_shape))


class RandomBanditEnvironment(bte.BanditTFEnvironment):
  """Bandit environment that returns random observations and rewards."""

  def __init__(self,
               observation_distribution: types.Distribution,
               reward_distribution: types.Distribution,
               action_spec: Optional[types.TensorSpec] = None):
    """Initializes an environment that returns random observations and rewards.

    Note that `observation_distribution` and `reward_distribution` are expected
    to have batch rank 1. That is, `observation_distribution.batch_shape` should
    have length exactly 1. `tensorflow_probability.distributions.Independent` is
    useful for manipulating batch and event shapes. For example,

    ```python
    observation_distribution = tfd.Independent(tfd.Normal(tf.zeros([12, 3, 4]),
                                                          tf.ones([12, 3, 4])))
    env = RandomBanditEnvironment(observation_distribution, ...)
    env.observation_spec  # tensor_spec.TensorSpec(shape=[3, 4], ...)
    env.batch_size  # 12
    ```

    Args:
      observation_distribution: a `tensorflow_probability.Distribution`.
        Batches of observations will be drawn from this distribution. The
        `batch_shape` of this distribution must have length 1 and be the same as
        the `batch_shape` of `reward_distribution`.
      reward_distribution: a `tensorflow_probability.Distribution`.
        Batches of rewards will be drawn from this distribution. The
        `batch_shape` of this distribution must have length 1 and be the same as
        the `batch_shape` of `observation_distribution`.
      action_spec: a `TensorSpec` describing the expected action. Note that
        actions are ignored and do not affect rewards.
    """
    observation_batch_shape = observation_distribution.batch_shape
    reward_batch_shape = reward_distribution.batch_shape
    reward_event_shape = reward_distribution.event_shape

    if observation_batch_shape.rank != 1:
      _raise_batch_shape_error(
          'observation_distribution', observation_batch_shape)

    if reward_batch_shape.rank != 1:
      _raise_batch_shape_error(
          'reward_distribution', observation_batch_shape)

    if reward_event_shape.rank != 0:
      raise ValueError('`reward_distribution` must have event_shape (); '
                       'got {}'.format(reward_event_shape))

    if reward_distribution.dtype != tf.float32:
      raise ValueError('`reward_distribution` must have dtype float32; '
                       'got {}'.format(reward_distribution.float32))

    if observation_batch_shape[0] != reward_batch_shape[0]:
      raise ValueError(
          '`reward_distribution` and `observation_distribution` must have the '
          'same batch shape; got {} and {}'.format(
              reward_batch_shape, observation_batch_shape))
    batch_size = tf.compat.dimension_value(observation_batch_shape[0])
    self._observation_distribution = observation_distribution
    self._reward_distribution = reward_distribution
    observation_spec = tensor_spec.TensorSpec(
        shape=self._observation_distribution.event_shape,
        dtype=self._observation_distribution.dtype,
        name='observation_spec')
    time_step_spec = time_step.time_step_spec(observation_spec)
    super(RandomBanditEnvironment, self).__init__(time_step_spec=time_step_spec,
                                                  action_spec=action_spec,
                                                  batch_size=batch_size)

  def _apply_action(self, action: types.NestedTensor) -> types.NestedTensor:
    del action  # unused
    return self._reward_distribution.sample()

  def _observe(self) -> types.NestedTensor:
    return self._observation_distribution.sample()
