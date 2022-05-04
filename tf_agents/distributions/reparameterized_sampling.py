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

"""Helper function to do reparameterized sampling if the distributions supports it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_probability as tfp

from tf_agents.distributions import gumbel_softmax


def sample(distribution, reparam=False, **kwargs):
  """Sample from distribution either with reparameterized sampling or regular sampling.

  Args:
    distribution: A `tfp.distributions.Distribution` instance.
    reparam: Whether to use reparameterized sampling.
    **kwargs: Parameters to be passed to distribution's sample() fucntion.

  Returns:
  """
  if reparam:
    if (distribution.reparameterization_type !=
        tfp.distributions.FULLY_REPARAMETERIZED):
      raise ValueError('This distribution cannot be reparameterized'
                       ': {}'.format(distribution))
    else:
      return distribution.sample(**kwargs)
  else:
    if isinstance(distribution, gumbel_softmax.GumbelSoftmax):
      samples = distribution.sample(**kwargs)
      return distribution.convert_to_one_hot(samples)
    else:
      return distribution.sample(**kwargs)
