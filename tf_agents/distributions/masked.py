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

"""Define distributions for spaces where not all actions are valid."""
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp


class MaskedCategorical(tfp.distributions.Categorical):
  """A categorical distribution which supports masks per step.

  Masked values are replaced with -inf inside the logits. This means the values
  will never be sampled.

  When computing the log probability of a set of actions, each action is
  assigned a probability under each sample. _log_prob is modified to only return
  the probability of a sample under the distribution for the same timestep.

  TODO(ddohan): Integrate entropy calculation from cl/207017752
  """

  def __init__(self,
               logits,
               mask,
               probs=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               neg_inf=-1e10,
               name='MaskedCategorical'):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
        set of Categorical distributions. The first `N - 1` dimensions index
        into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
      mask: A boolean mask. False/0 values mean a position should be masked out.
      probs: Must be `None`. Required to conform with base
        class `tfp.distributions.Categorical`.
      dtype: The type of the event samples (default: int32).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      neg_inf: None or Float. Value used to mask out invalid positions. If None,
        use logits.dtype.min to get a large negative number.
        Otherwise use given value.
      name: Python `str` name prefixed to Ops created by this class.
    """
    logits = tf.convert_to_tensor(value=logits)
    mask = tf.convert_to_tensor(value=mask)
    self._mask = tf.cast(mask, tf.bool)  # Nonzero values are True
    self._neg_inf = neg_inf
    if probs is not None:
      raise ValueError('Must provide masked predictions as logits.'
                       ' Probs are accepted for API compatibility with '
                       ' Categorical distribution. Given `%s`.' % probs)

    if neg_inf is None:
      neg_inf = logits.dtype.min
    neg_inf = tf.cast(
        tf.fill(dims=tf.shape(input=logits), value=neg_inf), logits.dtype)
    logits = tf.compat.v2.where(self._mask, logits, neg_inf)

    super(MaskedCategorical, self).__init__(
        logits=logits,
        probs=None,
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  def _entropy(self):
    entropy = tf.nn.log_softmax(self.logits) * self.probs_parameter()
    # Replace the (potentially -inf) values with 0s before summing.
    entropy = tf.compat.v1.where(self._mask, entropy, tf.zeros_like(entropy))
    return -tf.reduce_sum(input_tensor=entropy, axis=-1)

  @property
  def mask(self):
    return self._mask

  @property
  def parameters(self):
    params = super(MaskedCategorical, self).parameters
    params['mask'] = self.mask
    return params
