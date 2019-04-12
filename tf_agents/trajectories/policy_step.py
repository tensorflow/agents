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

"""Policy Step."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


# Returned with every call to policy.action() and policy.distribution().
#
# Attributes:
#   action: An action tensor or action distribution.
#   state: State of the policy to be fed back into the next call to
#     policy.action() or policy.distribution(), e.g. an RNN state. For stateless
#     policies, this will be the empty tuple.
#   info: Auxiliary information emitted by the policy, e.g. log probabilities of
#     the actions.
PolicyStep = collections.namedtuple('PolicyStep',
                                    ('action', 'state', 'info'))

# Set default empty tuple for PolicyStep.state and PolicyStep.info.
PolicyStep.__new__.__defaults__ = ((),) * len(PolicyStep._fields)
