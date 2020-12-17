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

"""Converts TensorFlow Policies into Python Policies."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text
from absl import logging

import tensorflow as tf
from tf_agents.policies import py_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import session_utils


class PyTFPolicy(py_policy.PyPolicy, session_utils.SessionUser):
  """Exposes a Python policy as wrapper over a TF Policy."""

  _time_step = ...  # type: ts.TimeStep
  _policy_state = ...  # type: types.NestedPlaceHolder
  _action_step = ...  # type: policy_step.PolicyStep

  # TODO(damienv): currently, the initial policy state must be batched
  # if batch_size is given. Without losing too much generality, the initial
  # policy state could be the same for every element in the batch.
  # In that case, the initial policy state could be given with no batch
  # dimension.
  # TODO(sfishman): Remove batch_size param entirely.
  def __init__(self,
               policy: tf_policy.TFPolicy,
               batch_size: Optional[int] = None,
               seed: Optional[types.Seed] = None):
    """Initializes a new `PyTFPolicy`.

    Args:
      policy: A TF Policy implementing `tf_policy.TFPolicy`.
      batch_size: (deprecated)
      seed: Seed to use if policy performs random actions (optional).
    """
    if not isinstance(policy, tf_policy.TFPolicy):
      logging.warning('Policy should implement tf_policy.TFPolicy')

    if batch_size is not None:
      logging.warning('In PyTFPolicy constructor, `batch_size` is deprecated, '
                      'this parameter has no effect. This argument will be '
                      'removed on 2019-05-01')

    time_step_spec = tensor_spec.to_nest_array_spec(policy.time_step_spec)
    action_spec = tensor_spec.to_nest_array_spec(policy.action_spec)
    super(PyTFPolicy, self).__init__(
        time_step_spec, action_spec, policy_state_spec=(), info_spec=())

    self._tf_policy = policy
    self.session = None

    self._policy_state_spec = tensor_spec.to_nest_array_spec(
        self._tf_policy.policy_state_spec)

    self._batch_size = None
    self._batched = None
    self._seed = seed
    self._built = False

  def _construct(self, batch_size, graph):
    """Construct the agent graph through placeholders."""

    self._batch_size = batch_size
    self._batched = batch_size is not None

    outer_dims = [self._batch_size] if self._batched else [1]
    with graph.as_default():
      self._time_step = tensor_spec.to_nest_placeholder(
          self._tf_policy.time_step_spec, outer_dims=outer_dims)
      self._tf_initial_state = self._tf_policy.get_initial_state(
          batch_size=self._batch_size or 1)

      self._policy_state = tf.nest.map_structure(
          lambda ps: tf.compat.v1.placeholder(  # pylint: disable=g-long-lambda
              ps.dtype,
              ps.shape,
              name='policy_state'),
          self._tf_initial_state)
      self._action_step = self._tf_policy.action(
          self._time_step, self._policy_state, seed=self._seed)

  def initialize(self,
                 batch_size: Optional[int],
                 graph: Optional[tf.Graph] = None):
    if self._built:
      raise RuntimeError('PyTFPolicy can only be initialized once.')

    if not graph:
      graph = tf.compat.v1.get_default_graph()

    self._construct(batch_size, graph)
    var_list = tf.nest.flatten(self._tf_policy.variables())
    common.initialize_uninitialized_variables(self.session, var_list)
    self._built = True

  def save(self,
           policy_dir: Optional[Text] = None,
           graph: Optional[tf.Graph] = None):
    if not self._built:
      raise RuntimeError('PyTFPolicy has not been initialized yet.')

    if not graph:
      graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      policy_checkpointer = common.Checkpointer(
          ckpt_dir=policy_dir, policy=self._tf_policy, global_step=global_step)
      policy_checkpointer.initialize_or_restore(self.session)
      with self.session.as_default():
        policy_checkpointer.save(global_step)

  def restore(self,
              policy_dir: Text,
              graph: Optional[tf.Graph] = None,
              assert_consumed: bool = True):
    """Restores the policy from the checkpoint.

    Args:
      policy_dir: Directory with the checkpoint.
      graph: A graph, inside which policy the is restored (optional).
      assert_consumed: If true, contents of the checkpoint will be checked
        for a match against graph variables.

    Returns:
      step: Global step associated with the restored policy checkpoint.

    Raises:
      RuntimeError: if the policy is not initialized.
      AssertionError: if the checkpoint contains variables which do not have
        matching names in the graph, and assert_consumed is set to True.

    """

    if not self._built:
      raise RuntimeError(
          'PyTFPolicy must be initialized before being restored.')
    if not graph:
      graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
      global_step = tf.compat.v1.train.get_or_create_global_step()
      policy_checkpointer = common.Checkpointer(
          ckpt_dir=policy_dir, policy=self._tf_policy, global_step=global_step)
      status = policy_checkpointer.initialize_or_restore(self.session)
      with self.session.as_default():
        if assert_consumed:
          status.assert_consumed()
        status.run_restore_ops()
      return self.session.run(global_step)

  def _build_from_time_step(self, time_step):
    outer_shape = nest_utils.get_outer_array_shape(time_step,
                                                   self._time_step_spec)
    if len(outer_shape) == 1:
      self.initialize(outer_shape[0])
    elif not outer_shape:
      self.initialize(None)
    else:
      raise ValueError(
          'Cannot handle more than one outer dimension. Saw {} outer '
          'dimensions: {}'.format(len(outer_shape), outer_shape))

  def _get_initial_state(self, batch_size):
    if not self._built:
      self.initialize(batch_size)
    if batch_size != self._batch_size:
      raise ValueError(
          '`batch_size` argument is different from the batch size provided '
          'previously. Expected {}, but saw {}.'.format(self._batch_size,
                                                        batch_size))
    return self.session.run(self._tf_initial_state)

  def _action(self, time_step, policy_state, seed: Optional[types.Seed] = None):
    if seed is not None:
      raise ValueError('`seed` is passed to the class as an argument.')
    if not self._built:
      self._build_from_time_step(time_step)

    batch_size = None
    if time_step.step_type.shape:
      batch_size = time_step.step_type.shape[0]
    if self._batch_size != batch_size:
      raise ValueError(
          'The batch size of time_step is different from the batch size '
          'provided previously. Expected {}, but saw {}.'.format(
              self._batch_size, batch_size))

    if not self._batched:
      # Since policy_state is given in a batched form from the policy and we
      # simply have to send it back we do not need to worry about it. Only
      # update time_step.
      time_step = nest_utils.batch_nested_array(time_step)

    nest_utils.assert_same_structure(self._time_step, time_step)
    feed_dict = {self._time_step: time_step}
    if policy_state is not None:
      # Flatten policy_state to handle specs that are not hashable due to lists.
      for state_ph, state in zip(
          tf.nest.flatten(self._policy_state), tf.nest.flatten(policy_state)):
        feed_dict[state_ph] = state

    action_step = self.session.run(self._action_step, feed_dict)
    action, state, info = action_step

    if not self._batched:
      action, info = nest_utils.unbatch_nested_array([action, info])

    return policy_step.PolicyStep(action, state, info)
