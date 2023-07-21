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

"""Gumbel_Softmax distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


class GumbelSoftmax(
    tfp.distributions.relaxed_onehot_categorical.RelaxedOneHotCategorical):
  """GumbelSoftmax distribution with temperature and logits.

  The implementation is almost identical to tfp.distributions.
  relaxed_onehot_categorical.RelaxedOneHotCategorical except for the following:

  1. Add mode() function to return mode of the underlying categorical
     distribution (There is no mode() defined in RelaxedOneHotCategorical)
  2. Add a convert_to_integer() function to convert the sample from non-integer
     to integer. Note that the sample function returns one_hot format of the
     discrete action that is different from regular distributions.
  3. log_prob() of RelaxedOneHotCategorical will return INF when the input is
     at boundary. In this implementation, we add a small epsilon to avoid
     getting NAN. In addition, when the input is discrete, we calculate log_prob
     using the underlying categorical distribution.

  """

  def __init__(
      self,
      temperature,
      logits=None,
      probs=None,
      dtype=tf.int32,
      validate_args=False,
      allow_nan_stats=True,
      name='GumbelSoftmax'):
    """Initialize GumbelSoftmax using class log-probabilities.

    Args:
      temperature: A `Tensor`, representing the temperature of one or more
        distributions. The temperature values must be positive, and the shape
        must broadcast against `(logits or probs)[..., 0]`.
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
        of one or many distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of logits for each class. Only one of `logits` or `probs` should
        be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of one or many distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of probabilities for each class. Only one of `logits` or `probs`
        should be passed in.
      dtype: The type of the event samples (default: int32).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    super(GumbelSoftmax, self).__init__(
        temperature=temperature,
        logits=logits,
        probs=probs,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats)

    self._output_dtype = dtype

  def _log_prob(self, x):
    if x.dtype != self.distribution.logits.dtype:
      # Calculate log_prob using the underlying categorical distribution when
      # the input is discrete.
      x = tf.cast(x, self.distribution.logits.dtype)
      return tf.reduce_sum(
          x * tf.math.log_softmax(self.distribution.logits), axis=-1)
    # Add an epsilon to prevent INF.
    x += 1e-10
    return super(GumbelSoftmax, self)._log_prob(x)

  def convert_to_one_hot(self, samples):
    return tf.one_hot(
        tf.argmax(samples, axis=-1),
        self.distribution.event_size, dtype=self._output_dtype)

  def _mode(self):
    return self.convert_to_one_hot(self.distribution.logits)


