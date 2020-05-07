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

# Lint as: python3
"""Multiprocessing hooks for TF-Agents."""

import multiprocessing as _multiprocessing

from typing import Text

_INITIALIZED = [False]
_INTERACTIVE = [False]

_NOT_INITIALIZED_ERROR = """Unable to load multiprocessing context.

Please ensure that you properly initialize your program by wrapping your main()
call:

def main(argv):
  ...

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(main)

or, if using absl.app:

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(lambda _: absl.app.run(main))


For unit tests, this also means wrapping your test.main using handle_test_main:

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_test_main(tf.test.main)

or

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_test_main(
      tf_agents.utils.test_utils.main)

If you are in interactive mode (e.g. python console, ipython, jupyter notebook)
use:

tf_agents.system.multiprocessing.enable_interactive_mode()
"""


def get_context(method: Text = None) -> _multiprocessing.context.BaseContext:
  """Get a context: an object with the same API as multiprocessing module.

  Args:
    method: (Optional.) The method name; a safe default is used.

  Returns:
    A multiprocessing context.

  Raises:
    RuntimeError: If main() was not executed via handle_main().
  """
  if not _INITIALIZED[0]:
    raise RuntimeError(_NOT_INITIALIZED_ERROR)
  return _multiprocessing.get_context(method)


def handle_main(parent_main_fn, *args, **kwargs):
  _INITIALIZED[0] = True
  return parent_main_fn(*args, **kwargs)


def handle_test_main(parent_main_fn, *args, **kwargs):
  _INITIALIZED[0] = True
  return parent_main_fn(*args, **kwargs)


def enable_interactive_mode():
  _INITIALIZED[0] = True
