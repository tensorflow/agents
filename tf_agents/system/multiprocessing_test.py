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

"""Tests for tf_agents.system.multiprocessing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import test_utils

_XVAL = 1


def get_xval(queue):
  global _XVAL
  queue.put(_XVAL)


class XValStateSaver(multiprocessing.StateSaver):

  def collect_state(self):
    global _XVAL
    return _XVAL

  def restore_state(self, state):
    global _XVAL
    _XVAL = state


def execute_pickled_fn(ra, queue):
  pickle.loads(ra)(queue)


class MultiprocessingTest(test_utils.TestCase):

  def testGinBindingsInOtherProcess(self):
    # Serialize a function that we will call in subprocesses
    serialized_get_xval = pickle.dumps(get_xval)

    # get_xval accesses _XVAL, we set the state to 2 and will check that
    # subprocesses will see this value.
    global _XVAL
    _XVAL = 2

    ctx = multiprocessing.get_context()

    # Local function should easily access _XVAL
    local_queue = ctx.SimpleQueue()
    execute_pickled_fn(serialized_get_xval, local_queue)
    self.assertFalse(local_queue.empty())
    self.assertEqual(local_queue.get(), 2)

    # Remote function can access new _XVAL since part of running it
    # is serializing the state via XValStateSaver (passed to handle_test_main
    # below).
    remote_queue = ctx.SimpleQueue()
    p = ctx.Process(
        target=execute_pickled_fn, args=(serialized_get_xval, remote_queue))
    p.start()
    p.join()
    self.assertFalse(remote_queue.empty())
    self.assertEqual(remote_queue.get(), 2)

  def testPool(self):
    ctx = multiprocessing.get_context()
    p = ctx.Pool(3)
    x = 1
    values = p.map(x.__add__, [3, 4, 5, 6, 6])
    self.assertEqual(values, [4, 5, 6, 7, 7])

  def testArgExpected(self):
    no_argument_main_fn = lambda: None
    with self.assertRaises(TypeError):
      multiprocessing.handle_main(no_argument_main_fn)


if __name__ == '__main__':
  multiprocessing.handle_test_main(
      test_utils.main, extra_state_savers=[XValStateSaver()])
