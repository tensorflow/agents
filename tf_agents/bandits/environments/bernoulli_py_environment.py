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

"""Class implementation of Python Bernoulli Bandit environment."""
# Using Type Annotations.
from typing import Optional, Sequence

import gin
import numpy as np

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.specs import array_spec
from tf_agents.typing import types


@gin.configurable
class BernoulliPyEnvironment(bandit_py_environment.BanditPyEnvironment):
  """Implements finite-armed Bernoulli Bandits.

  This environment implements a finite-armed non-contextual Bernoulli Bandit
  environment as a subclass of BanditPyEnvironment. For every arm, the reward
  distribution is 0/1 (Bernoulli) with parameter p set at the initialization.
  For a reference, see e.g., Example 1.1 in "A Tutorial on Thompson Sampling" by
  Russo et al. (https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
  """

  def __init__(self,
               means: Sequence[types.Float],
               batch_size: Optional[types.Int] = 1):
    """Initializes a Bernoulli Bandit environment.

    Args:
      means: vector of floats in [0, 1], the mean rewards for actions. The
        number of arms is determined by its length.
      batch_size: (int) The batch size.
    """
    self._num_actions = len(means)
    self._means = means
    self._batch_size = batch_size
    if any(x < 0 or x > 1 for x in means):
      raise ValueError('All parameters should be floats in [0, 1].')

    action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=self._num_actions - 1,
        name='action')
    observation_spec = array_spec.ArraySpec(
        shape=(1,), dtype=np.int32, name='observation')
    super(BernoulliPyEnvironment, self).__init__(observation_spec, action_spec)

  def _observe(self) -> types.NestedArray:
    return np.zeros(
        shape=[self._batch_size] + list(self.observation_spec().shape),
        dtype=self.observation_spec().dtype)

  def _apply_action(self, action: types.NestedArray) -> types.Float:
    return [np.floor(self._means[i] + np.random.random()) for i in action]

  @property
  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> types.Int:
    return self._batch_size
