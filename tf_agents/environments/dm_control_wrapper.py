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

"""Wrapper providing a PyEnvironmentBase adapter for Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

_as_float32_array = functools.partial(np.asarray, dtype=np.float32)


def _maybe_float32(o):
  if o.dtype == np.float64:
    return _as_float32_array(o)
  return o


def convert_time_step(time_step):
  """Convert to agents time_step type as the __hash__ method is different."""
  reward = time_step.reward
  if reward is None:
    reward = 0.0
  discount = time_step.discount
  if discount is None:
    discount = 1.0

  observation = tf.nest.map_structure(_maybe_float32, time_step.observation)
  return ts.TimeStep(
      ts.StepType(time_step.step_type),
      _as_float32_array(reward),
      _as_float32_array(discount),
      observation,
  )


def convert_spec(spec):
  if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
    tfa_spec = array_spec.BoundedArraySpec.from_spec(spec)
  else:
    tfa_spec = array_spec.ArraySpec.from_spec(spec)

  if tfa_spec.dtype == np.float64:
    tfa_spec = tfa_spec.replace(dtype=np.float32)
  return tfa_spec


class DmControlWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Base wrapper forwarding the DM control types into the tf_agents ones."""

  def __init__(self, env, render_kwargs=None):
    super(DmControlWrapper, self).__init__(env)
    render_kwargs = render_kwargs or {}
    self._render_kwargs = render_kwargs

    self._observation_spec = tf.nest.map_structure(convert_spec,
                                                   self._env.observation_spec())
    self._action_spec = tf.nest.map_structure(convert_spec,
                                              self._env.action_spec())

  @property
  def physics(self):
    return self._env.physics

  def _reset(self):
    return convert_time_step(self._env.reset())

  def _step(self, action):
    action = tf.nest.map_structure(lambda a, s: np.asarray(a, dtype=s.dtype),
                                   action, self._env.action_spec())
    return convert_time_step(self._env.step(action))

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def close(self):
    self._env.close()

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Only rgb_array rendering mode is supported. Got %s' %
                       mode)
    return self._env.physics.render(**self._render_kwargs)
