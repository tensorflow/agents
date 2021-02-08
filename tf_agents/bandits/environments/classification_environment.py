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

"""An environment based on an arbitrary classification problem."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.environments import bandit_tf_environment as bte
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.typing import types
from tf_agents.utils import eager_utils


tfd = tfp.distributions


def _batched_table_lookup(tbl, row, col):
  """Mapped 2D table lookup.

  Args:
    tbl: a `Tensor` of shape `[r, s, t]`.
    row: a `Tensor` of dtype `int32` with shape `[r]` and values in
      the range `[0, s - 1]`.
    col: a `Tensor` of dtype `int32` with shape `[r]` and values in
      the range `[0, t - 1]`.
  Returns:
    A `Tensor` `x` with shape `[r]` where `x[i] = tbl[i, row[i], col[i]`.
  """
  assert_correct_shapes = tf.group(
      tf.assert_equal(tf.shape(row), tf.shape(col)),
      tf.assert_equal(tf.shape(row)[0], tf.shape(tbl)[0]))
  rng = tf.range(tf.shape(row)[0])
  idx = tf.stack([rng, row, col], axis=-1)
  with tf.control_dependencies([assert_correct_shapes]):
    values = tf.gather_nd(tbl, idx)
  return values


@gin.configurable
class ClassificationBanditEnvironment(bte.BanditTFEnvironment):
  """An environment based on an arbitrary classification problem."""

  def __init__(self,
               dataset: tf.data.Dataset,
               reward_distribution: types.Distribution,
               batch_size: types.Int,
               label_dtype_cast: Optional[tf.DType] = None,
               shuffle_buffer_size: Optional[types.Int] = None,
               repeat_dataset: Optional[bool] = True,
               prefetch_size: Optional[types.Int] = None,
               seed: Optional[types.Int] = None,
               name: Optional[Text] = 'classification'):
    """Initialize `ClassificationBanditEnvironment`.

    Args:
      dataset: a `tf.data.Dataset` consisting of two `Tensor`s, [inputs, labels]
        where inputs can be of any shape, while labels are integer class labels.
        The label tensor can be of any rank as long as it has 1 element.
      reward_distribution: a `tfd.Distribution` with event_shape
        `[num_classes, num_actions]`. Entry `[i, j]` is the reward for taking
        action `j` for an instance of class `i`.
      batch_size: if `dataset` is batched, this is the size of the batches.
      label_dtype_cast: if not None, casts dataset labels to this dtype.
      shuffle_buffer_size: If None, do not shuffle.  Otherwise, a shuffle buffer
        of the specified size is used in the environment's `dataset`.
      repeat_dataset: Makes the environment iterate on the `dataset` once
        avoiding `OutOfRangeError:  End of sequence` errors when the environment
        is stepped past the end of the `dataset`.
      prefetch_size: If None, do not prefetch.  Otherwise, a prefetch buffer
        of the specified size is used in the environment's `dataset`.
      seed: Used to make results deterministic.
      name: The name of this environment instance.
    Raises:
      ValueError: if `reward_distribution` does not have an event shape with
        rank 2.
    """

    # Computing `action_spec`.
    event_shape = reward_distribution.event_shape
    if len(event_shape) != 2:
      raise ValueError(
          'reward_distribution must have event shape of rank 2; '
          'got event shape {}'.format(event_shape))
    _, num_actions = event_shape
    action_spec = tensor_spec.BoundedTensorSpec(shape=(),
                                                dtype=tf.int32,
                                                minimum=0,
                                                maximum=num_actions - 1,
                                                name='action')
    output_shapes = tf.compat.v1.data.get_output_shapes(dataset)

    # Computing `time_step_spec`.
    if len(output_shapes) != 2:
      raise ValueError('Dataset must have exactly two outputs; got {}'.format(
          len(output_shapes)))
    context_shape = output_shapes[0]
    context_dtype, lbl_dtype = tf.compat.v1.data.get_output_types(dataset)
    if label_dtype_cast:
      lbl_dtype = label_dtype_cast
    observation_spec = tensor_spec.TensorSpec(
        shape=context_shape, dtype=context_dtype)
    time_step_spec = time_step.time_step_spec(observation_spec)

    super(ClassificationBanditEnvironment, self).__init__(
        action_spec=action_spec,
        time_step_spec=time_step_spec,
        batch_size=batch_size,
        name=name)

    if shuffle_buffer_size:
      dataset = dataset.shuffle(buffer_size=shuffle_buffer_size,
                                seed=seed,
                                reshuffle_each_iteration=True)
    if repeat_dataset:
      dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if prefetch_size:
      dataset = dataset.prefetch(prefetch_size)
    self._data_iterator = eager_utils.dataset_iterator(dataset)
    self._current_label = tf.compat.v2.Variable(
        tf.zeros(batch_size, dtype=lbl_dtype))
    self._previous_label = tf.compat.v2.Variable(
        tf.zeros(batch_size, dtype=lbl_dtype))
    self._reward_distribution = reward_distribution
    self._label_dtype = lbl_dtype

    reward_means = self._reward_distribution.mean()
    self._optimal_action_table = tf.argmax(
        reward_means, axis=1, output_type=self._action_spec.dtype)
    self._optimal_reward_table = tf.reduce_max(reward_means, axis=1)

  def _observe(self) -> types.NestedTensor:
    context, lbl = eager_utils.get_next(self._data_iterator)
    self._previous_label.assign(self._current_label)
    self._current_label.assign(tf.reshape(
        tf.cast(lbl, dtype=self._label_dtype), shape=[self._batch_size]))
    return tf.reshape(
        context,
        shape=[self._batch_size] + self._time_step_spec.observation.shape)

  def _apply_action(self, action: types.NestedTensor) -> types.NestedTensor:
    action = tf.reshape(
        action, shape=[self._batch_size] + self._action_spec.shape)
    reward_samples = self._reward_distribution.sample(tf.shape(action))
    return _batched_table_lookup(reward_samples, self._current_label, action)

  def compute_optimal_action(self) -> types.NestedTensor:
    return tf.gather(
        params=self._optimal_action_table, indices=self._previous_label)

  def compute_optimal_reward(self) -> types.NestedTensor:
    return tf.gather(
        params=self._optimal_reward_table, indices=self._previous_label)
