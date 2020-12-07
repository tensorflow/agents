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
"""Async helper for the policy saver."""

import threading
from typing import Text
from absl import logging

from tf_agents.policies import policy_saver as policy_saver_module


class AsyncPolicySaver(object):
  """Triggers `policy_saver` save calls in a separate thread asynchronously."""

  def __init__(self, policy_saver: policy_saver_module.PolicySaver):
    """Initialize an AsyncPolicySaver.

    Args:
      policy_saver: An instance of a `policy_saver.PolicySaver`.
    """
    self._policy_saver = policy_saver
    self._save_condition_variable = threading.Condition()

    # These vars should only be accessed if the lock in save_condition is held.
    # export_dir is set to None whenever there is no pending save. Otherwise it
    # is used to communicate across threads.
    self._export_dir = None
    self._saving_checkpoint = False
    self._join_save_thread = False

    self._save_thread = threading.Thread(target=self._save_loop)
    self._save_thread.start()

  def _save_loop(self):
    """Helper method for the saving thread to wait and execute save requests."""
    while True:
      with self._save_condition_variable:
        while not self._export_dir:
          self._save_condition_variable.wait()
          if self._join_save_thread:
            return
        if self._saving_checkpoint:
          logging.info("Saving checkpoint to %s", self._export_dir)
          self._policy_saver.save_checkpoint(self._export_dir)
        else:
          logging.info("Saving policy to %s", self._export_dir)
          self._policy_saver.save(self._export_dir)
        self._export_dir = None
        self._save_condition_variable.notify()

  def _assert_save_thread_is_alive(self):
    if self._join_save_thread or not self._save_thread.is_alive():
      raise ValueError("Saving thread in AsyncPolicySaver is not alive. Either "
                       "an exception has occured while saving, or the saver "
                       "was closed.")

  def save(self, export_dir: Text, blocking: bool = False):
    """Triggers an async save of the policy to the given `export_dir`.

    Only one save can be triggered at a time. If `save` or `save_checkpoint`
    are called while another save of either kind is still ongoing the saving is
    skipped.

    If blocking is set then the call will block until any ongoing saves finish,
    and then a new save will be made before returning.

    Args:
      export_dir: Directory path for the `saved_model` of the policy.
      blocking: If True the call to save will block until a save can be
        performed and finished. If a save was ongoing it will wait for that to
        finish, and then do a blocking save before returning.
    """
    self._save(export_dir, saving_checkpoint=False, blocking=blocking)

  def save_checkpoint(self, export_dir: Text, blocking: bool = False):
    """Triggers an async save of the policy checkpoint.

    Only one save can be triggered at a time. If `save` or `save_checkpoint`
    are called while another save of either kind is still ongoing the saving is
    skipped.

    If blocking is set then the call will block until any ongoing saves finish,
    and then a new save will be made before returning.

    Args:
      export_dir: Directory path for the checkpoint of the policy.
      blocking: If True the call to save will block until a save can be
        performed and finished. If a save was ongoing it will wait for that to
        finish, and then do a blocking save before returning.
    """
    self._save(export_dir, saving_checkpoint=True, blocking=blocking)

  def _save(self, export_dir, saving_checkpoint, blocking):
    """Helper save method, generalizes over save and save_checkpoint."""
    self._assert_save_thread_is_alive()

    if blocking:
      with self._save_condition_variable:
        while self._export_dir:
          logging.info("Waiting for AsyncPolicySaver to finish.")
          self._save_condition_variable.wait()
        if saving_checkpoint:
          self._policy_saver.save_checkpoint(export_dir)
        else:
          self._policy_saver.save(export_dir)
      return

    if not self._save_condition_variable.acquire(blocking=False):
      logging.info("AsyncPolicySaver save is still in progress skipping save.")
      return
    try:
      self._saving_checkpoint = saving_checkpoint
      self._export_dir = export_dir
      self._save_condition_variable.notify()
    finally:
      self._save_condition_variable.release()

  def flush(self):
    """Blocks until there is no saving happening."""
    with self._save_condition_variable:
      while self._export_dir:
        logging.info("Waiting for AsyncPolicySaver to finish.")
        self._save_condition_variable.wait()

  def close(self):
    """Blocks until there is no saving happening and kills the save_thread."""
    with self._save_condition_variable:
      while self._export_dir:
        logging.info("Waiting for AsyncPolicySaver to finish.")
        self._save_condition_variable.wait()
      self._join_save_thread = True
      self._save_condition_variable.notify()
    self._save_thread.join()
