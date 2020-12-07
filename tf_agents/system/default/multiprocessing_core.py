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

# Lint as: python3
"""Multiprocessing hooks for TF-Agents."""

import abc
import multiprocessing as _multiprocessing

from typing import Any, Text

from absl import app

__all__ = [
    'StateSaver',
    'handle_main',
    'handle_test_main',
    'enable_interactive_mode',
]

_INITIALIZED = [False]
_INTERACTIVE = [False]
_STATE_SAVERS = []


def initialized():
  return _INITIALIZED[0]


class StateSaver(object):
  """Class for getting and setting global state."""

  @abc.abstractmethod
  def collect_state(self) -> Any:
    pass

  @abc.abstractmethod
  def restore_state(self, state: Any) -> None:
    pass


def get_context(method: Text = None) -> _multiprocessing.context.BaseContext:
  return _multiprocessing.get_context(method)


def handle_main(parent_main_fn, *args, **kwargs):
  """Function that wraps the main function in a multiprocessing-friendly way.

  This function additionally accepts an `extra_state_savers` kwarg;
  users can provide a list of `tf_agents.multiprocessing.StateSaver` instances,
  where a `StateSaver` tells multiprocessing how to store some global state
  and how to restore it in the subprocess.

  Args:
    parent_main_fn: A callable.
    *args: rgs for `parent_main_fn`.
    **kwargs: kwargs for `parent_main_fn`.
      This may also include `extra_state_savers` kwarg.

  Returns:
    Output of `parent_main_fn`.
  """
  extra_state_savers = kwargs.pop('extra_state_savers', [])
  _STATE_SAVERS.extend(extra_state_savers)
  _INITIALIZED[0] = True
  return app.run(parent_main_fn, *args, **kwargs)


def handle_test_main(parent_main_fn, *args, **kwargs):
  """Function that wraps the test main in a multiprocessing-friendly way.

  This function additionally accepts an `extra_state_savers` kwarg;
  users can provide a list of `tf_agents.multiprocessing.StateSaver` instances,
  where a `StateSaver` tells multiprocessing how to store some global state
  and how to restore it in the subprocess.

  Args:
    parent_main_fn: A callable.
    *args: rgs for `parent_main_fn`.
    **kwargs: kwargs for `parent_main_fn`.
      This may also include `extra_state_savers` kwarg.

  Returns:
    Output of `parent_main_fn`.
  """
  extra_state_savers = kwargs.pop('extra_state_savers', [])
  _STATE_SAVERS.extend(extra_state_savers)
  _INITIALIZED[0] = True
  return parent_main_fn(*args, **kwargs)


def enable_interactive_mode(extra_state_savers=None):
  """Function that enables multiprocessing in interactive mode.

  This function accepts an `extra_state_savers` argument;
  users can provide a list of `tf_agents.multiprocessing.StateSaver` instances,
  where a `StateSaver` tells multiprocessing how to store some global state
  and how to restore it in the subprocess.

  Args:
    extra_state_savers: A list of `StateSaver` instances.
  """
  if _INITIALIZED[0]:
    raise ValueError('Multiprocessing already initialized')
  extra_state_savers = extra_state_savers or []
  _STATE_SAVERS.extend(extra_state_savers)
  _INITIALIZED[0] = True
