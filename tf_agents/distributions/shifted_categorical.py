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

"""Categorical distribution where values are shifted to honor a range."""

import tensorflow as tf
import tensorflow_probability as tfp


class ShiftedCategorical(tfp.distributions.Categorical):
  """Categorical distribution with support [shift, shift + K] instead of [0, K].

  Simply a thin wrapper around Categorical which takes a user provided minimal
  value and shifts the minimum value using the provided value. This distribution
  allows policies where the user provides an action_spec range, e.g. QPolicy, to
  honor it, by offsetting the sampled value.
  """

  def __init__(self,
               logits=None,
               probs=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               shift=None,
               name="ShiftedCategorical"):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
        set of Categorical distributions. The first `N - 1` dimensions index
        into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of a set of Categorical distributions. The first `N - 1` dimensions
        index into a batch of independent distributions and the last dimension
        represents a vector of probabilities for each class. Only one of
        `logits` or `probs` should be passed in.
      dtype: The type of the event samples (default: int32).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      shift: value to shift the interval such that the sampled values are
        between [shift, shift + K] instead of [0, K].
      name: Python `str` name prefixed to Ops created by this class.
    """
    if shift is None:
      raise ValueError("ShiftedCategorical expects a shift value.""")

    self._shift = shift
    super(ShiftedCategorical, self).__init__(
        logits=logits,
        probs=probs,
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  def log_prob(self, value, name="log_prob"):
    """Log probability density/mass function."""
    value -= self._shift
    return super(ShiftedCategorical, self).log_prob(value, name)

  def prob(self, value, name="prob"):
    """Probability density/mass function."""
    value -= self._shift
    return super(ShiftedCategorical, self).prob(value, name)

  def cdf(self, value, name="log_cdf"):
    """Cumulative distribution function."""
    value -= self._shift
    return super(ShiftedCategorical, self).cdf(value, name)

  def log_cdf(self, value, name="log_cdf"):
    """Log cumulative distribution function."""
    value -= self._shift
    return super(ShiftedCategorical, self).log_cdf(value, name)

  def mode(self, name="mode"):
    """Mode of the distribution."""
    mode = super(ShiftedCategorical, self).mode(name)
    return mode + self._shift

  def sample(self, sample_shape=(), seed=None, name="sample", **kwargs):
    """Generate samples of the specified shape."""
    sample = super(ShiftedCategorical, self).sample(
        sample_shape=sample_shape, seed=seed, name=name, **kwargs)
    return sample + self._shift

  @property
  def shift(self):
    return self._shift

  @property
  def parameters(self):
    params = super(ShiftedCategorical, self).parameters
    params["shift"] = self._shift
    return params
