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

"""Utility environment that creates random observations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import tf_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class RandomTFEnvironment(tf_environment.TFEnvironment):
  """Randomly generates observations following the given observation_spec.

  If an action_spec is provided, it validates that the actions used to step the
  environment are compatible with the given spec.
  """

  def __init__(self,
               time_step_spec,
               action_spec,
               batch_size=1,
               episode_end_probability=0.1):
    """Initializes the environment.

    Args:
      time_step_spec: A `TimeStep` namedtuple containing `TensorSpec`s defining
        the Tensors returned by `step()` (step_type, reward, discount, and
        observation).
      action_spec: A nest of BoundedTensorSpec representing the actions of the
        environment.
      batch_size: The batch size expected for the actions and observations.
      episode_end_probability: Probability an episode will end when the
        environment is stepped.
    """
    super(RandomTFEnvironment, self).__init__(
        time_step_spec, action_spec, batch_size=batch_size)
    self._episode_end_probability = episode_end_probability

    def _variable_from_spec(name, spec):
      full_shape = [batch_size] + spec.shape.as_list()
      if not name:
        name = "spec_var"
      return common.create_variable(name, 0, shape=full_shape, dtype=spec.dtype)

    paths_and_specs = nest_utils.flatten_with_joined_paths(time_step_spec)
    variables = [
        _variable_from_spec(path, spec) for path, spec in paths_and_specs
    ]
    self._time_step_variables = tf.nest.pack_sequence_as(
        time_step_spec, variables)

  def _current_time_step(self):
    """Returns the current `TimeStep`."""
    return tf.nest.map_structure(tf.identity, self._time_step_variables)

  def _update_time_step(self, time_step):
    tf.nest.map_structure(lambda var, value: var.assign(value),
                          self._time_step_variables, time_step)

  def _sample_obs_and_reward(self):
    sampled_observation = tensor_spec.sample_spec_nest(
        self._time_step_spec.observation, outer_dims=(self.batch_size,))
    sampled_reward = tensor_spec.sample_spec_nest(
        self._time_step_spec.reward, outer_dims=(self.batch_size,))
    return sampled_observation, sampled_reward

  @common.function
  def _reset(self):
    """Resets the environment and returns the current time_step."""
    obs, _ = self._sample_obs_and_reward()
    time_step = ts.restart(
        obs, self._batch_size, reward_spec=self._time_step_spec.reward)
    self._update_time_step(time_step)
    return self._current_time_step()

  @common.function(autograph=True)
  def _step(self, action):
    """Steps the environment according to the action."""
    # Make sure the given action is compatible with the spec. We compare it to
    # t[0] as the spec doesn't have a batch dim.
    tf.nest.map_structure(
        lambda spec, t: tf.Assert(spec.is_compatible_with(t[0]), [t]),
        self._action_spec, action)

    # If we generalize the batched data to not terminate at the same time, we
    # will need to only reset the correct batch_inidices.
    if self._time_step_variables.is_last()[0]:
      return self.reset()

    obs, reward = self._sample_obs_and_reward()
    # Note: everything in the batch terminates at the same time.
    if tf.random.uniform(()) < self._episode_end_probability:
      time_step = ts.termination(obs, reward)
    else:
      time_step = ts.transition(obs, reward)

    self._update_time_step(time_step)
    return time_step
