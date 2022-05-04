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

"""Actions sampler and sample clipper interface.

N: number of samples
B: batch size
A: action size

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class ActionsSampler(object):
  """Base class for actions sampler.

  Given a batch of distribution params(including mean and var), sample, clip and
  return [N, B, A] actions, where 'N' means num_samples, 'B' means batch_size,
  'A' means action_size.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, action_spec, sample_clippers=None, sample_rejecters=None):
    """Creates an ActionsSampler.

    Args:
      action_spec: A nest of BoundedTensorSpec representing the actions.
      sample_clippers: A list of callables that are applied to
        the generated samples. These callables take in a nested structure
        matching the action_spec and must return a matching structure.
      sample_rejecters: A list of callables that will reject samples and return
        a mask tensor.
    """
    self._action_spec = action_spec
    self._sample_clippers = sample_clippers
    self._sample_rejecters = sample_rejecters

  @abc.abstractmethod
  def refit_distribution_to(self, target_sample_indices, samples):
    """Refits distribution according to actions with index of ind.

    Args:
      target_sample_indices: A [B, M] sized tensor indicating the index
      samples: A nested structure corresponding to action_spec. Each action is
        a [B, N, A] sized tensor.

    Returns:
      distribution_params: Distribution related parameters refitted to
        best samples.
    """
    raise NotImplementedError('refit_distribution not implemented.')

  @abc.abstractmethod
  def sample_batch_and_clip(
      self,
      num_samples,
      distribution_params,
      state=None):
    """Samples and clips a batch of actions [N, B, A] with distribution params.

    Args:
      num_samples: Number of actions to sample each round.
      distribution_params: Distribution related parameters. The sampler will use
        it to sample actions.
      state: Nested state tensor constructed according to oberservation_spec
        of the task.

    Returns:
      actions: batch containing the full action vector
    """
    raise NotImplementedError('sample_batch_and_clip not implemented.')


class SampleClipper(object):
  """Base class for sampler clipper.

  Given a batch of actions, clip and return clipped actions according to given
  constraints.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __call__(self, actions, state):
    """Clips action according to given constraints.

    Args:
      actions: An [N, B, A] Tensor representing sampled actions.
      state: Nested state tensor.
    Returns:
      actions: An [N, B, A] Tensor representing clipped actions.
    """
    raise NotImplementedError('clip not implemented.')


class SampleRejecter(object):
  """Base class for sampler clipper.

  Given a batch of actions, clip and return clipped actions according to given
  constraints.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __call__(self, actions, state):
    """Clips action according to given constraints.

    Args:
      actions: An [N, B, A] Tensor representing sampled actions.
      state: Nested state tensor.
    Returns:
      actions: An [N, B, A] Tensor representing clipped actions.
    """
    raise NotImplementedError('clip not implemented.')

