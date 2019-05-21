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

"""Utilities related to distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.distributions import tanh_bijector_stable
from tf_agents.utils import common


def scale_distribution_to_spec(distribution, spec):
  """Scales the given distribution to the bounds of the given spec."""
  return SquashToSpecNormal(distribution, spec)


class SquashToSpecNormal(tfp.distributions.Distribution):
  """Scales an input normalized action distribution to match spec bounds.

  Unlike the normal distribution computed when NormalProjectionNetwork
  is called with scale_distribution=False, which merely squashes the mean
  of the distribution to within the action spec, this distribution scales the
  output distribution to ensure that the output action fits within the spec.

  This distribution also maintains the input normal distribution, and uses this
  distribution to compute the KL-divergence between two SquashToSpecNormal
  distributions provided that they were scaled by the same action spec.
  This is possible as KL divergence is invariant when both distributions are
  transformed using the same invertible function.

  Formally, let a be the action magnitude and b be the action mean. The
  squashing operation performs the following change of variables to the
  input distribution X:

  Y = a * tanh(X) + b

  Note that this is a change of variables as the function is invertible, with:

  X = tan((Y - b) / a), where Y in (b - a, b + a)
  """

  def __init__(self,
               distribution,
               spec,
               validate_args=False,
               name="SquashToSpecNormal"):
    """Constructs a SquashToSpecNormal distribution.

    Args:
      distribution: input normal distribution with normalized mean and std dev
      spec: bounded action spec from which to compute action ranges
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.
    """

    if not isinstance(distribution, tfp.distributions.Normal):
      raise ValueError("Input distribution must be a normal distribution, "
                       "got {} instead".format(distribution))
    self.action_means, self.action_magnitudes = common.spec_means_and_magnitudes(
        spec)
    # Parameters here describe the actor network's output, which is a normalized
    # distribution prior to squashing to the action spec.
    # This is necessary (and sufficient) in order for policy info to compare an
    # old policy to a new policy.
    parameters = {"loc": distribution.loc, "scale": distribution.scale}
    # The raw action distribution
    self.input_distribution = distribution

    bijectors = [
        tfp.bijectors.AffineScalar(
            shift=self.action_means, scale=self.action_magnitudes),
        tanh_bijector_stable.Tanh()
    ]
    bijector_chain = tfp.bijectors.Chain(bijectors)
    self._squashed_distribution = tfp.distributions.TransformedDistribution(
        distribution=distribution, bijector=bijector_chain)
    super(SquashToSpecNormal, self).__init__(
        dtype=distribution.dtype,
        reparameterization_type=distribution.reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=distribution.allow_nan_stats,
        parameters=parameters,
        # We let TransformedDistribution access _graph_parents since this class
        # is more like a baseclass than derived.
        graph_parents=(
            distribution._graph_parents +  # pylint: disable=protected-access
            bijector_chain.graph_parents),
        name=name)

  def kl_divergence(self, other, name="kl_divergence"):
    """Computes the KL Divergence between two SquashToSpecNormal distributions."""
    if not isinstance(other, SquashToSpecNormal):
      raise ValueError("other distribution should be of type "
                       "SquashToSpecNormal, got {}".format(other))
    if (tf.reduce_any(tf.not_equal(self.action_means, other.action_means)) or
        tf.reduce_any(
            tf.not_equal(self.action_magnitudes, other.action_magnitudes))):
      raise ValueError("Other distribution does not have same action mean "
                       "and magnitude. This mean {}, this magnitude {}, "
                       "other mean {}, other magnitude {}.".format(
                           self.action_means, self.action_magnitudes,
                           other.action_means, other.action_magnitudes))
    return self.input_distribution.kl_divergence(other.input_distribution, name)

  def sample(self, sample_shape=(), seed=None, name="sample"):
    """Generates samples from the wrapped TransformedDistribution."""
    return self._squashed_distribution.sample(sample_shape, seed, name)

  def log_prob(self, value, name="log_prob"):
    """Computes log probability from the wrapped TransformedDistribution."""
    return self._squashed_distribution.log_prob(value, name)

  def mode(self, name="mode"):
    """Compute mean of the SquashToSpecNormal distribution."""
    mean = self.action_magnitudes * tf.tanh(self.input_distribution.mode()) + \
        self.action_means
    return mean
