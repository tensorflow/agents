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

"""Wrappers for TF environments.

Use tf_agents.environments.wrapper for PyEnvironments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import tf_environment
from tf_agents.specs import tensor_spec


class TFEnvironmentBaseWrapper(tf_environment.TFEnvironment):
  """Base class for TFEnvrionment wrappers."""

  def __init__(self, env):
    super(TFEnvironmentBaseWrapper, self).__init__()
    self._env = env

  def __getattr__(self, name):
    if name in self.__dict__:
      return getattr(self, name)
    return getattr(self._env, name)

  def time_step_spec(self):
    return self._env.time_step_spec()

  def action_spec(self):
    return self._env.action_spec()

  def observation_spec(self):
    return self._env.observation_spec()

  @property
  def batched(self):
    return self._env.batched

  @property
  def batch_size(self):
    return self._env.batch_size

  def _current_time_step(self):
    return self._env.current_time_step()

  def _reset(self):
    return self._env.reset()

  def _step(self, action):
    return self._env.step(action)

  def render(self):
    return self._env.render()


class OneHotActionWrapper(TFEnvironmentBaseWrapper):
  """Converts discrete action to one_hot format."""

  def __init__(self, env):
    super(OneHotActionWrapper, self).__init__(env)
    self._validate_action_spec()

  def _validate_action_spec(self):

    def _validate(action_spec):
      if action_spec.dtype.is_integer and len(action_spec.shape.as_list()) > 1:
        raise ValueError(
            'OneHotActionWrapper only supports actions with at most one '
            'dimension! action_spec: {}'.format(action_spec))

    tf.nest.map_structure(_validate, self._env.action_spec())

  def action_spec(self):

    def convert_to_one_hot(action_spec):
      """Convert action_spec to one_hot format."""
      if action_spec.dtype.is_integer:
        num_actions = action_spec.maximum - action_spec.minimum + 1
        output_shape = action_spec.shape + (num_actions,)

        return tensor_spec.BoundedTensorSpec(
            shape=output_shape,
            dtype=action_spec.dtype,
            minimum=0,
            maximum=1,
            name='one_hot_action_spec')
      else:
        return action_spec

    return tf.nest.map_structure(convert_to_one_hot, self._env.action_spec())

  def _step(self, action):
    action = tf.argmax(action, axis=-1, output_type=action.dtype)
    return self._env.step(action)
