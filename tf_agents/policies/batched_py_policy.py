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
"""Treat multiple non-batch policies as a single batch policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=line-too-long
# multiprocessing.dummy provides a pure *multithreaded* threadpool that works
# in both python2 and python3 (concurrent.futures isn't available in python2).
#   https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.dummy
from multiprocessing import dummy as mp_threads
# pylint: enable=line-too-long

from typing import Sequence, Optional

import gin

from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step as ps
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


@gin.configurable
class BatchedPyPolicy(py_policy.PyPolicy):
  """Batch together multiple py policies and act as a single batch.

  The policies should only access shared python variables using
  shared mutex locks (from the threading module).
  """

  def __init__(self,
               policies: Sequence[py_policy.PyPolicy],
               multithreading: bool = True):
    """Batch together multiple (non-batched) py policies.

    The policies can be different but must use the same action and
    observation specs.

    Args:
      policies: List python policies (must be non-batched).
      multithreading: Python bool describing whether interactions with the
        given policies should happen in their own threadpool.  If `False`,
        then all interaction is performed serially in the current thread.

        This may be combined with `TFPyPolicy(..., py_policy_is_batched=True)`
        to ensure that multiple policies are all run in the same thread.

    Raises:
      ValueError: If policies is not a list or tuple, or is zero length, or if
        one of the policies is already batched.
      ValueError: If the action or observation specs don't match.
    """
    if not isinstance(policies, (list, tuple)):
      raise ValueError("policies must be a list or tuple.  Got: %s" % policies)

    self._parallel_execution = multithreading
    self._policies = policies
    self._num_policies = len(policies)
    self._time_step_spec = self._policies[0].time_step_spec
    self._action_spec = self._policies[0].action_spec
    self._policy_state_spec = self._policies[0].policy_state_spec
    self._info_spec = self._policies[0].info_spec
    self._policy_step_spec = self._policies[0].policy_step_spec
    self._trajectory_spec = self._policies[0].trajectory_spec
    self._collect_data_spec = self._policies[0].collect_data_spec
    self._observation_and_action_constraint_splitter = \
        self._policies[0].observation_and_action_constraint_splitter

    self._validate_spec(py_policy.PyPolicy.time_step_spec,
                        self._time_step_spec)
    self._validate_spec(py_policy.PyPolicy.action_spec,
                        self._action_spec)
    self._validate_spec(py_policy.PyPolicy.policy_state_spec,
                        self._policy_state_spec)
    self._validate_spec(py_policy.PyPolicy.info_spec,
                        self._info_spec)
    self._validate_spec(py_policy.PyPolicy.policy_step_spec,
                        self._policy_step_spec)
    self._validate_spec(py_policy.PyPolicy.trajectory_spec,
                        self._trajectory_spec)
    self._validate_spec(py_policy.PyPolicy.collect_data_spec,
                        self._collect_data_spec)
    self._validate_spec(
        py_policy.PyPolicy.observation_and_action_constraint_splitter,
        self._observation_and_action_constraint_splitter)

    # Create a multiprocessing threadpool for execution.
    if multithreading:
      self._pool = mp_threads.Pool(self._num_policies)

    super(BatchedPyPolicy, self).__init__(
        self._time_step_spec,
        self._action_spec,
        self._policy_state_spec,
        self._info_spec,
        self._observation_and_action_constraint_splitter)

  def __del__(self):
    """Join external processes, if necessary."""
    if self._parallel_execution:  # pytype: disable=attribute-error  # trace-all-classes
      self._pool.close()  # pytype: disable=attribute-error  # trace-all-classes
      self._pool.join()  # pytype: disable=attribute-error  # trace-all-classes

  def _validate_spec(self, policy_spec_method, spec_to_match):
    # pytype: disable=attribute-error
    if any(policy_spec_method.__get__(p) != spec_to_match
           for p in self._policies):
      raise ValueError(
          "All policies must have the same specs.  Saw: %s" % self._policies)
    # pytype: enable=attribute-error

  def _execute(self, fn, iterable):
    if self._parallel_execution:  # pytype: disable=attribute-error  # trace-all-classes
      return self._pool.map(fn, iterable)  # pytype: disable=attribute-error  # trace-all-classes
    else:
      return [fn(x) for x in iterable]

  def _get_initial_state(self, batch_size: int) -> types.NestedArray:
    if self._num_policies == 1:  # pytype: disable=attribute-error  # trace-all-classes
      return nest_utils.batch_nested_array(
          self._policies[0].get_initial_state())  # pytype: disable=attribute-error  # trace-all-classes
    else:
      infos = self._execute(_execute_get_initial_state, self._policies)  # pytype: disable=attribute-error  # trace-all-classes
      infos = nest_utils.unbatch_nested_array(infos)
      return nest_utils.stack_nested_arrays(infos)

  def _action(self,
              time_step: ts.TimeStep,
              policy_state: types.NestedArray,
              seed: Optional[types.Seed] = None) -> ps.PolicyStep:
    """Forward a batch of time_step and policy_states to the wrapped policies.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: An Array, or a nested dict, list or tuple of Arrays
        representing the previous policy_state.
      seed: Seed value used to initialize a pseudorandom number generator.

    Returns:
      A batch of `PolicyStep` named tuples, each one containing:
        `action`: A nest of action Arrays matching the `action_spec()`.
        `state`: A nest of policy states to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.

    Raises:
      NotImplementedError: if `seed` is not None.
    """
    if seed is not None:
      raise NotImplementedError(
          "seed is not supported; but saw seed: {}".format(seed))
    if self._num_policies == 1:  # pytype: disable=attribute-error  # trace-all-classes
      time_step = nest_utils.unbatch_nested_array(time_step)
      policy_state = nest_utils.unbatch_nested_array(policy_state)
      policy_steps = self._policies[0].action(time_step, policy_state)  # pytype: disable=attribute-error  # trace-all-classes
      return nest_utils.batch_nested_array(policy_steps)
    else:
      unstacked_time_steps = nest_utils.unstack_nested_arrays(time_step)
      if len(unstacked_time_steps) != len(self._policies):  # pytype: disable=attribute-error  # trace-all-classes
        raise ValueError(
            "Primary dimension of time_step items does not match "
            "batch size: %d vs. %d" % (len(unstacked_time_steps),
                                       len(self._policies)))  # pytype: disable=attribute-error  # trace-all-classes
      unstacked_policy_states = [()] * len(unstacked_time_steps)
      if policy_state:
        unstacked_policy_states = nest_utils.unstack_nested_arrays(policy_state)
        if len(unstacked_policy_states) != len(self._policies):  # pytype: disable=attribute-error  # trace-all-classes
          raise ValueError(
              "Primary dimension of policy_state items does not match "
              "batch size: %d vs. %d" % (len(unstacked_policy_states),
                                         len(self._policies)))  # pytype: disable=attribute-error  # trace-all-classes
      policy_steps = self._execute(_execute_policy,
                                   zip(self._policies,  # pytype: disable=attribute-error  # trace-all-classes
                                       unstacked_time_steps,
                                       unstacked_policy_states))
      return nest_utils.stack_nested_arrays(policy_steps)


def _execute_policy(zip_results_element) -> ps.PolicyStep:
  """Called on each element of zip return value, in _action method."""
  (policy, time_step, policy_state) = zip_results_element
  return policy.action(time_step, policy_state)


def _execute_get_initial_state(policy) -> types.NestedArray:
  """Called on each policy in _get_initial_state method."""
  return policy.get_initial_state(batch_size=1)
