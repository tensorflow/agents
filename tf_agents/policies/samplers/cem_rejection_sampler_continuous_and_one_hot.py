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

"""Actions sampler that supports rejection sampling of the following type of actions.

See the base class for more information.
"""

import gin
import tensorflow as tf

from tf_agents.policies.samplers import cem_actions_sampler_continuous_and_one_hot


@gin.configurable
class GaussianActionsSampler(
    cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler):
  """Hybrid actions sampler that samples continuous actions using Gaussian distribution.

  See the base class for more information.
  This class supports sampling through sample_validator. The sample_validator
  needs to be a callable. By wrapping with tf.numpy_function, it can support
  complex env interaction when deciding whether to reject/resample inside it.
  """

  def __init__(self, action_spec, sample_validator, sub_actions_fields=None):

    super(GaussianActionsSampler, self).__init__(
        action_spec, sub_actions_fields, sub_actions_fields)

    self._sample_validator = sample_validator

  def _sample_continuous_and_transpose(
      self, mean, var, state, i, one_hot_index):
    num_samples_continuous = self._number_samples_all[i]

    def sample_and_transpose(mean, var, spec, mask, sample_validator):
      if spec.dtype.is_integer:
        sample = tf.one_hot(one_hot_index, self._num_mutually_exclusive_actions)
        sample = tf.broadcast_to(
            sample,
            [tf.shape(mean)[0],
             tf.constant(num_samples_continuous),
             tf.shape(mean)[1]])
      else:
        sample = sample_validator(
            num_samples_continuous, mean, var, state)
        sample = tf.transpose(sample, [1, 0, 2])
        sample = sample * mask
      return tf.cast(sample, spec.dtype)

    # [B, N, A]
    samples_continuous = tf.nest.map_structure(
        sample_and_transpose, mean, var, self._action_spec, self._masks[i],
        self._sample_validator)

    return samples_continuous
