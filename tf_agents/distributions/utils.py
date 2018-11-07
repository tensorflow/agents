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

"""Utilities for generating and working with distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.distributions import layers

nest = tf.contrib.framework.nest


def project_to_output_distribution(inputs,
                                   output_spec,
                                   project_to_discrete,
                                   project_to_continuous,
                                   outer_rank=1,
                                   scope='project_to_output'):
  """Project a batch of inputs to a distribution object.

  Args:
    inputs: An input Tensor of shape [batch_size, None].
    output_spec: A single output spec.
    project_to_discrete: The method to use for projecting a discrete output.
    project_to_continuous: The method to use for projecting a continuous output.
    outer_rank: The number of outer dimensions of inputs to consider batch
      dimensions and to treat as batch dimensions of output distribution.
    scope: The variable scope.

  Returns:
    A distribution object corresponding to the arguments and output spec
      provided.

  Raises:
    ValueError: If the distribution type of output_spec is unclear.
  """
  with tf.variable_scope(scope):
    if output_spec.is_discrete():
      return project_to_discrete(inputs, output_spec, outer_rank=outer_rank)
    elif output_spec.is_continuous():
      return project_to_continuous(inputs, output_spec, outer_rank=outer_rank)
    else:
      raise ValueError('Output spec corresponds to unknown distribution.')


def project_to_output_distributions(
    inputs,
    output_spec,
    project_to_discrete=layers.factored_categorical,
    project_to_continuous=layers.normal,
    outer_rank=1,):
  """Project a batch of inputs to distribution objects.

  Args:
    inputs: An input Tensor of shape [batch_size, None].
    output_spec: A possibly nested tuple of output specs.
    project_to_discrete: The method to use for projecting a discrete output.
    project_to_continuous: The method to use for projecting a continuous output.
    outer_rank: The number of outer dimensions of inputs to consider batch
      dimensions and to treat as batch dimensions of output distribution.
  Returns:
    A possibly nested tuple of the same structure as output_spec where each
      element is the distribution for the corresponding output.
  """
  flat_output_spec = nest.flatten(output_spec)
  flat_distributions = []
  for i, single_output_spec in enumerate(flat_output_spec):
    dist = project_to_output_distribution(
        inputs,
        single_output_spec,
        project_to_discrete=project_to_discrete,
        project_to_continuous=project_to_continuous,
        outer_rank=outer_rank,
        scope='output%d' % i)
    flat_distributions.append(dist)

  return nest.pack_sequence_as(output_spec, flat_distributions)
