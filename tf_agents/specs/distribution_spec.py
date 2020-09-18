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

"""Spec definition for tensorflow_probability.Distribution."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import tensorflow_probability as tfp

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions


class DistributionSpec(object):
  """Describes a tfp.distribution.Distribution."""

  __slots__ = [
      "_builder", "_input_params_spec", "_sample_spec",
      "_distribution_parameters"
  ]

  def __init__(self, builder, input_params_spec, sample_spec,
               **distribution_parameters):
    """Creates a DistributionSpec.

    Args:
      builder: Callable function(**params) which returns a Distribution
        following the spec.
      input_params_spec: Nest of tensor_specs describing the tensor parameters
        required for building the described distribution.
      sample_spec: Data type of the output samples of the described
        distribution.
      **distribution_parameters: Extra parameters for building the distribution.
    """
    self._builder = builder
    self._input_params_spec = input_params_spec
    self._sample_spec = sample_spec
    self._distribution_parameters = distribution_parameters

  @property
  def builder(self):
    """Returns the `distribution_builder` of the spec."""
    return self._builder

  @property
  def input_params_spec(self):
    """Returns the `input_params_spec` of the spec."""
    return self._input_params_spec

  @property
  def sample_spec(self):
    """Returns the `sample_spec` of the spec."""
    return self._sample_spec

  @property
  def distribution_parameters(self):
    """Returns the `distribution_parameters` of the spec."""
    return self._distribution_parameters

  def build_distribution(self, **distribution_parameters):
    """Creates an instance of the described distribution.

    The spec's paramers are updated with the given ones.
    Args:
      **distribution_parameters: Kwargs update the spec's distribution
        parameters.

    Returns:
      Distribution instance.
    """
    kwargs = self._distribution_parameters.copy()
    kwargs.update(distribution_parameters)
    return self._builder(**kwargs)

  def __repr__(self):
    return ("DistributionSpec(builder={}, input_params_spec={}, "
            "sample_spec={})").format(self.builder,
                                      repr(self.input_params_spec),
                                      repr(self.sample_spec))


def deterministic_distribution_from_spec(spec):
  """Creates a Deterministic distribution_spec from a tensor_spec."""
  return DistributionSpec(tfd.Deterministic, {"loc": spec}, sample_spec=spec)


def nested_distributions_from_specs(specs, parameters):
  """Builds a nest of distributions from a nest of specs.

  Args:
    specs: A nest of distribution specs.
    parameters: A nest of distribution kwargs.

  Returns:
    Nest of distribution instances with the same structure as the given specs.
  """
  return nest.map_structure_up_to(
      specs, lambda spec, parameters: spec.build_distribution(**parameters),
      specs, parameters)
