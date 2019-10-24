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

"""Environment implementation that generates random observations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class RandomPyEnvironment(py_environment.PyEnvironment):
  """Randomly generates observations following the given observation_spec.

  If an action_spec is provided it validates that the actions used to step the
  environment fall within the defined spec.
  """

  def __init__(self,
               observation_spec,
               action_spec=None,
               episode_end_probability=0.1,
               discount=1.0,
               reward_fn=None,
               batch_size=None,
               seed=42,
               render_size=(2, 2, 3),
               min_duration=0,
               max_duration=None):
    """Initializes the environment.

    Args:
      observation_spec: An `ArraySpec`, or a nested dict, list or tuple of
        `ArraySpec`s.
      action_spec: An `ArraySpec`, or a nested dict, list or tuple of
        `ArraySpec`s.
      episode_end_probability: Probability an episode will end when the
        environment is stepped.
      discount: Discount to set in time_steps.
      reward_fn: Callable that takes in step_type, action, an observation(s),
        and returns a numpy array of rewards.
      batch_size: (Optional) Number of observations generated per call.
        If this value is not `None`, then all actions are expected to
        have an additional major axis of size `batch_size`, and all outputs
        will have an additional major axis of size `batch_size`.
      seed: Seed to use for rng used in observation generation.
      render_size: Size of the random render image to return when calling
        render.
      min_duration: Number of steps at the beginning of the
        episode during which the episode can not terminate.
      max_duration: Optional number of steps after which the episode
        terminates regarless of the termination probability.

    Raises:
      ValueError: If batch_size argument is not None and does not match the
      shapes of discount or reward.
    """
    self._batch_size = batch_size
    self._observation_spec = observation_spec
    self._time_step_spec = ts.time_step_spec(self._observation_spec)
    self._action_spec = action_spec or []
    self._episode_end_probability = episode_end_probability
    discount = np.asarray(discount, dtype=np.float32)

    if self._batch_size:
      if not discount.shape:
        discount = np.tile(discount, self._batch_size)
      if self._batch_size != len(discount):
        raise ValueError('Size of discounts must equal the batch size.')
    self._discount = discount

    if reward_fn is None:
      # Return a reward whose size matches the batch size
      if self._batch_size is None:
        self._reward_fn = lambda *_: np.asarray(0.0, dtype=np.float32)
      else:
        self._reward_fn = (
            lambda *_: np.zeros(self._batch_size, dtype=np.float32))
    else:
      self._reward_fn = reward_fn

    self._done = True
    self._num_steps = 0
    self._min_duration = min_duration
    self._max_duration = max_duration
    self._rng = np.random.RandomState(seed)
    self._render_size = render_size
    super(RandomPyEnvironment, self).__init__()

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def batched(self):
    return False if self._batch_size is None else True

  def _get_observation(self):
    batch_size = (self._batch_size,) if self._batch_size else ()
    return array_spec.sample_spec_nest(self._observation_spec, self._rng,
                                       batch_size)

  def _reset(self):
    self._done = False
    return ts.restart(self._get_observation(), self._batch_size)

  def _check_reward_shape(self, reward):
    expected_shape = () if self._batch_size is None else (self._batch_size,)
    if np.asarray(reward).shape != expected_shape:
      raise ValueError('%r != %r. Size of reward must equal the batch size.' %
                       (np.asarray(reward).shape, self._batch_size))

  def _step(self, action):
    if self._done:
      return self.reset()

    if self._action_spec:
      tf.nest.assert_same_structure(self._action_spec, action)

    self._num_steps += 1

    observation = self._get_observation()
    if self._num_steps < self._min_duration:
      self._done = False
    elif self._max_duration and self._num_steps >= self._max_duration:
      self._done = True
    else:
      self._done = self._rng.uniform() < self._episode_end_probability

    if self._done:
      reward = self._reward_fn(ts.StepType.LAST, action, observation)
      self._check_reward_shape(reward)
      time_step = ts.termination(observation, reward)
      self._num_steps = 0
    else:
      reward = self._reward_fn(ts.StepType.MID, action, observation)
      self._check_reward_shape(reward)
      time_step = ts.transition(observation, reward, self._discount)

    return time_step

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError(
          "Only rendering mode supported is 'rgb_array', got {} instead.".
          format(mode))

    return self._rng.randint(0, 256, size=self._render_size, dtype=np.uint8)

  def seed(self, seed):
    self._rng.seed(seed)
