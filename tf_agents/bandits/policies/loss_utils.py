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

"""Loss utility code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def pinball_loss(
    y_true, y_pred, weights=1.0, scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    quantile=0.5):
  """Adds a Pinball loss for quantile regression.

    ```
  loss = quantile * (y_true - y_pred)         if y_true > y_pred
  loss = (quantile - 1) * (y_true - y_pred)   otherwise
  ```

  See: https://en.wikipedia.org/wiki/Quantile_regression#Quantiles


  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
    quantile: A float between 0. and 1., the quantile we want to regress.

  Returns:
    Weighted Pinball loss float `Tensor`. If `reduction` is `NONE`, this has the
    same shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if y_true is None:
    raise ValueError('y_true must not be None.')
  if y_pred is None:
    raise ValueError('y_pred must not be None.')
  with tf.compat.v1.name_scope(scope, 'pinball_loss',
                               (y_pred, y_true, weights)) as scope:
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    error = tf.subtract(y_true, y_pred)
    loss_tensor = tf.maximum(quantile * error, (quantile - 1) * error)
    return tf.compat.v1.losses.compute_weighted_loss(
        loss_tensor, weights, scope, loss_collection, reduction=reduction)
