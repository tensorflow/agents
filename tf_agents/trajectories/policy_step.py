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
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import Any, Mapping, Text, NamedTuple, Optional, Union
from tf_agents.typing import types


ActionType = Union[types.NestedSpecTensorOrArray, types.NestedDistribution]


class PolicyStep(
    NamedTuple(
        'PolicyStep',
        [('action', ActionType),
         ('state', types.NestedSpecTensorOrArray),
         ('info', types.NestedSpecTensorOrArray)
        ])):
  """Returned with every call to `policy.action()` and `policy.distribution()`.

  Attributes:
   action: An action tensor or action distribution for `TFPolicy`, or numpy
     array for `PyPolicy`.
   state: State of the policy to be fed back into the next call to
     policy.action() or policy.distribution(), e.g. an RNN state. For stateless
     policies, this will be an empty tuple.
   info: Auxiliary information emitted by the policy, e.g. log probabilities of
     the actions. For policies without info this will be an empty tuple.
  """
  __slots__ = ()

  def replace(self, **kwargs) -> 'PolicyStep':
    """Exposes as namedtuple._replace.

    Usage:
    ```
      new_policy_step = policy_step.replace(action=())
    ```

    This returns a new policy step with an empty action.

    Args:
      **kwargs: key/value pairs of fields in the policy step.

    Returns:
      A new `PolicyStep`.
    """
    return self._replace(**kwargs)


# Set default empty tuple for PolicyStep.state and PolicyStep.info.
PolicyStep.__new__.__defaults__ = ((),) * len(PolicyStep._fields)


class CommonFields(object):
  """Strings which can be used for querying returned PolicyStep.info field.

  For example, use getattr(info, CommonFields.LOG_PROBABILITY, None) to check if
  log probabilities are returned in the step or not.
  """
  LOG_PROBABILITY = 'log_probability'


# Generic PolicyInfo object which is recommended to be subclassed when requiring
# that log-probabilities are returned, but having a custom namedtuple instead.
PolicyInfo = collections.namedtuple('PolicyInfo',
                                    (CommonFields.LOG_PROBABILITY,))


def _maybe_set_value_namedtuple_or_dict(obj: Any, key: Text, value: Any) -> Any:
  if isinstance(obj, Mapping):
    obj[key] = value
    return obj
  if getattr(obj, '_fields', None) is not None:
    return obj._replace(**{key: value})
  return obj


def _maybe_get_value_namedtuple_or_dict(
    obj: Any, key: Text, default_value: Any) -> Any:
  if isinstance(obj, Mapping):
    return obj.get(key, default_value)
  if getattr(obj, '_fields', None) is not None:
    return getattr(obj, key, default_value)
  return None


def set_log_probability(
    info: types.NestedTensorOrArray,
    log_probability: types.Float) -> types.NestedTensorOrArray:
  """Sets the CommonFields.LOG_PROBABILITY on info to be log_probability."""
  if info in ((), None):
    return PolicyInfo(log_probability=log_probability)
  return _maybe_set_value_namedtuple_or_dict(
      info, CommonFields.LOG_PROBABILITY, log_probability)


def get_log_probability(
    info: types.NestedTensorOrArray,
    default_log_probability: Optional[types.Float] = None) -> types.Float:
  """Gets the CommonFields.LOG_PROBABILITY from info depending on type."""
  return _maybe_get_value_namedtuple_or_dict(
      info, CommonFields.LOG_PROBABILITY, default_log_probability)
