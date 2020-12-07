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
"""Multiprocessing library for TF-Agents."""

import multiprocessing as _multiprocessing
from typing import Text

import cloudpickle
import gin
import gym


from tf_agents.system.default import multiprocessing_core
from tf_agents.system.default.multiprocessing_core import *  # pylint: disable=wildcard-import


_STATE_SAVERS = multiprocessing_core._STATE_SAVERS  # pylint: disable=protected-access


_NOT_INITIALIZED_ERROR = """Unable to load multiprocessing context.

Please ensure that you properly initialize your program by wrapping your main()
call:

def main(argv):
  ...

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(main, extra_state_savers=...)

or, if using absl.app:

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(
      functools.partial(absl.app.run, main), extra_state_savers=...)


For unit tests, this also means wrapping your test.main using handle_test_main:

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_test_main(
      tf.test.main, extra_state_savers=...)

or

if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_test_main(
      tf_agents.utils.test_utils.main, extra_state_savers=...)

If you are in interactive mode (e.g. python console, ipython, jupyter notebook)
use:

tf_agents.system.multiprocessing.enable_interactive_mode(
    extra_state_savers=...)

For more details on state savers, see the docstrings for
`tf_agents.multiprocessing.handle_*` and:

https://pythonspeed.com/articles/python-multiprocessing/
"""


def get_context(method: Text = None) -> _multiprocessing.context.BaseContext:
  """Get a context: an object with the same API as multiprocessing module.

  Args:
    method: (Optional.) The method name; a Google-safe default is provided.

  Returns:
    A multiprocessing context.

  Raises:
    RuntimeError: If main() was not executed via handle_main().
  """
  if not multiprocessing_core.initialized():
    raise RuntimeError(_NOT_INITIALIZED_ERROR)
  return _rewrite_target_with_state(multiprocessing_core.get_context(method))


class _WrappedTargetWithState:
  """Wraps a function to reload global state before it executes in a subprocess.

  The `__call__` method calls `target()` after reloading global state;
  it also logs any errors to stderr.  After creation, this object
  will be pickled by the multiprocessing module; and its __call__ method
  is executed in the subprocess.  We take great care to safely pickle
  all global state returned from the state savers (see the
  `extra_state_savers` arguments to handle_main, handle_test_main, etc).

  The `__call__` also captures exceptions and logs them to stderr of the
  main process.
  """

  def __init__(self, context, target):
    """Store target function and global state.

    This function runs on the process that's creating subprocesses.

    Args:
      context: An instance of a multiprocessing BaseContext.
      target: A callable that will be run in a subprocess.
    """
    self._context = context
    # Use cloudpickle to serialize target as this allows much more flexible
    # target functions, e.g., lambdas, to be passed to Process()/Pool().
    self._target = cloudpickle.dumps(target)
    self._global_state = []
    for saver in _STATE_SAVERS:
      try:
        self._global_state.append(cloudpickle.dumps(saver.collect_state()))
      except TypeError as e:
        context.get_logger().error(
            'Error while pickling global state from saver %s: %s.  Skipping.',
            saver, e)
        self._global_state.append(None)

  def __call__(self, *args, **kwargs):
    """Load global state and run target function.

    This function runs on the subprocess.

    Args:
      *args: Arguments to target.
      **kwargs: Keyword arguments to target.

    Returns:
      Return value of target.

    Raises:
      Reraises any exceptions by target.
    """
    try:
      if len(_STATE_SAVERS) != len(self._global_state):
        raise RuntimeError(
            'Expected number of state savers to match count of state values, '
            'but saw {} vs. {}'.format(len(_STATE_SAVERS), self._global_state))

      # Deserialize and restore global state
      for saver, state in zip(_STATE_SAVERS, self._global_state):
        if state is not None:
          saver.restore_state(cloudpickle.loads(state))

      # Perform the actual computation
      target = cloudpickle.loads(self._target)
      return target(*args, **kwargs)
    except Exception as e:
      logger = self._context.log_to_stderr()
      logger.error(e)
      raise e


def _rewrite_target_with_state(context):
  """Replaces context.Process.__init__ with a fn that stores global state."""
  wrapped_context = _ContextWrapper(context)
  return wrapped_context


# pylint: disable=invalid-name
class _PoolWrapper:
  """Wrapper for multiprocessing Pool that wraps function."""

  def __init__(self, context, pool):
    self._context = context
    self._pool = pool

  def apply(self, func, args=None, kwds=None):
    args = args or ()
    kwds = kwds or {}
    if func is not None:
      func = _WrappedTargetWithState(self._context, func)
    return self._pool.apply(func, args=args, kwds=kwds)

  def apply_async(self, func, args=None, kwds=None, callback=None,
                  error_callback=None):
    args = args or ()
    kwds = kwds or {}
    if func is not None:
      func = _WrappedTargetWithState(self._context, func)
    return self._pool.apply_async(func,
                                  args=args,
                                  kwds=kwds,
                                  callback=callback,
                                  error_callback=error_callback)

  def map(self, func, iterable, chunksize=None):
    if func is not None:
      func = _WrappedTargetWithState(self._context, func)
    return self._pool.map(func, iterable=iterable, chunksize=chunksize)

  def map_async(self, func, iterable, chunksize=None, callback=None,
                error_callback=None):
    if func is not None:
      func = _WrappedTargetWithState(self._context, func)
    return self._pool.map_async(func, iterable=iterable, chunksize=chunksize,
                                callback=callback,
                                error_callback=error_callback)

  def imap(self, *args, **kwargs):
    raise NotImplementedError('imap not implemented; try map')

  def imap_unordered(self, *args, **kwargs):
    raise NotImplementedError('imap_unordered not implemented; try map')

  def __getattr__(self, k):
    return getattr(self._pool, k)


class _ContextWrapper:
  """Wrapper for a multiprocessing Context that overrides Process and Pool."""

  def __init__(self, context):
    self._context = context

  def Process(self, group=None, target=None, name=None, args=None, kwargs=None,
              *, daemon=None):
    args = args or ()
    kwargs = kwargs or {}
    if target is not None:
      target = _WrappedTargetWithState(self._context, target)
    return self._context.Process(group=group, target=target, name=name,
                                 args=args, kwargs=kwargs, daemon=daemon)

  def Pool(self, processes=None, initializer=None, initargs=(),
           maxtasksperchild=None):
    return _PoolWrapper(
        self._context,
        self._context.Pool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild)
    )

  def __getattr__(self, k):
    return getattr(self._context, k)
# pylint: enable=invalid-name


class GinStateSaver(multiprocessing_core.StateSaver):
  """Sets and restores internal gin state."""

  def collect_state(self):
    return gin.config.config_str()

  def restore_state(self, state):
    gin.config.parse_config(state)


class OpenAIGymStateSaver(multiprocessing_core.StateSaver):
  """Sets and restores OpenAI gym registry."""

  def collect_state(self):
    return gym.envs.registration.registry

  def restore_state(self, state):
    if not isinstance(state, gym.envs.registration.EnvRegistry):
      raise RuntimeError(
          'Expected gym registry object of type {}, but saw state {}'
          .format(gym.envs.registration.EnvRegistry, state))
    gym.envs.registration.registry = state


_STATE_SAVERS.extend([
    GinStateSaver(),
    OpenAIGymStateSaver(),
])
