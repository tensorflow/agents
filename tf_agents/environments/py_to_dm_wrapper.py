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

"""Wrapper to convert a TF Agents Python environment into a DM Environment."""

from typing import Text

import dm_env
from dm_env import specs as dm_spec
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec as tfa_spec
from tf_agents.trajectories import time_step as ts
import tree


def _convert_timestep(timestep: ts.TimeStep) -> dm_env.TimeStep:
  if timestep.step_type.is_first():
    step_type = dm_env.StepType.FIRST
  elif timestep.step_type.is_last():
    step_type = dm_env.StepType.LAST
  elif timestep.step_type.is_mid():
    step_type = dm_env.StepType.MID
  else:
    raise ValueError(f'Invalid Step type: {timestep.step_type}')
  return dm_env.TimeStep(
      step_type=step_type,
      reward=None if step_type.first() else timestep.reward,
      discount=None if step_type.first() else timestep.discount,
      observation=timestep.observation,
  )


def _convert_spec(spec: tfa_spec.ArraySpec) -> dm_spec.Array:
  if isinstance(spec, tfa_spec.BoundedArraySpec):
    return dm_spec.BoundedArray(
        shape=spec.shape,
        dtype=spec.dtype,
        minimum=spec.minimum,
        maximum=spec.maximum,
        name=spec.name)
  else:
    return dm_spec.Array(shape=spec.shape, dtype=spec.dtype, name=spec.name)


class PyToDMWrapper(dm_env.Environment):
  """Environment wrapper to convert TF environments in DM environments."""

  def __init__(self, env: py_environment.PyEnvironment):
    super(PyToDMWrapper, self).__init__()
    self._environment = env
    if env.batched:
      raise NotImplementedError(
          'Batched environments cannot be converted to dm environments.')
    self._observation_spec = tree.map_structure(_convert_spec,
                                                env.observation_spec())
    self._action_spec = tree.map_structure(_convert_spec, env.action_spec())
    self._discount_spec = tree.map_structure(_convert_spec, env.discount_spec())

  def __getattr__(self, name: Text):
    """Forward all other calls to the base environment."""
    return getattr(self._environment, name)

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    return _convert_timestep(self._environment.reset())

  def step(self, action) -> dm_env.TimeStep:
    """Steps the environment."""
    return _convert_timestep(self._environment.step(action))

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def discount_spec(self) -> dm_spec.BoundedArray:
    return self._discount_spec

  @property
  def environment(self) -> py_environment.PyEnvironment:
    """Returns the wrapped environment."""
    return self._environment

  def close(self):
    self._environment.close()
