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

import tensorflow_probability as tfp
from tf_agents.distributions import tanh_bijector_stable
from tf_agents.utils import common


def scale_distribution_to_spec(distribution, spec):
  """Scales the given distribution to the bounds of the given spec."""
  bijectors = []

  # Bijector to rescale actions to ranges in action spec.
  action_means, action_magnitudes = common.spec_means_and_magnitudes(spec)
  bijectors.append(
      tfp.bijectors.AffineScalar(
          shift=action_means, scale=action_magnitudes))

  # Bijector to squash actions to range (-1.0, +1.0).
  bijectors.append(tanh_bijector_stable.Tanh())

  # Chain applies bijectors in reverse order, so squash will happen
  # before rescaling to action spec.
  bijector_chain = tfp.bijectors.Chain(bijectors)
  distributions = tfp.distributions.TransformedDistribution(
      distribution=distribution, bijector=bijector_chain)
  return distributions
