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

"""Policy implementation that generates random actions."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Sequence

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.distributions import masked
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class RandomPyPolicy(py_policy.PyPolicy):
  """Returns random samples of the given action_spec."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedArraySpec,
               info_spec: types.NestedArraySpec = (),
               seed: Optional[types.Seed] = None,
               outer_dims: Optional[Sequence[int]] = None,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None):
    """Initializes the RandomPyPolicy.

    Args:
      time_step_spec: Reference `time_step_spec`. If not None and outer_dims
        is not provided this is used to infer the outer_dims required for the
        given time_step when action is called.
      action_spec: A nest of BoundedArraySpec representing the actions to sample
        from.
      info_spec: Nest of `tf.TypeSpec` representing the data in the policy
        info field.
      seed: Optional seed used to instantiate a random number generator.
      outer_dims: An optional list/tuple specifying outer dimensions to add to
        the spec shape before sampling. If unspecified the outer_dims are
        derived from the outer_dims in the given observation when `action` is
        called.
      observation_and_action_constraint_splitter: A function used to process
        observations with action constraints. These constraints can indicate,
        for example, a mask of valid/invalid actions for a given state of the
        environment.
        The function takes in a full observation and returns a tuple consisting
        of 1) the part of the observation intended as input to the network and
        2) the constraint. An example
        `observation_and_action_constraint_splitter` could be as simple as:
        ```
        def observation_and_action_constraint_splitter(observation):
          return observation['network_input'], observation['constraint']
        ```
        *Note*: when using `observation_and_action_constraint_splitter`, make
        sure the provided `q_network` is compatible with the network-specific
        half of the output of the `observation_and_action_constraint_splitter`.
        In particular, `observation_and_action_constraint_splitter` will be
        called on the observation before passing to the network.
        If `observation_and_action_constraint_splitter` is None, action
        constraints are not applied.
    """
    self._seed = seed
    self._outer_dims = outer_dims

    if observation_and_action_constraint_splitter is not None:
      if not isinstance(action_spec, array_spec.BoundedArraySpec):
        raise NotImplementedError(
            'RandomPyPolicy only supports action constraints for '
            'BoundedArraySpec action specs.')

      scalar_shape = not action_spec.shape
      single_dim_shape = action_spec.shape == (1,) or action_spec.shape == [1]

      if not scalar_shape and not single_dim_shape:
        raise NotImplementedError(
            'RandomPyPolicy only supports action constraints for action specs '
            'shaped as () or (1,) or their equivalent list forms.')

    self._rng = np.random.RandomState(seed)
    if time_step_spec is None:
      time_step_spec = ts.time_step_spec()

    super(RandomPyPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter))

  def _action(self, time_step, policy_state, seed: Optional[types.Seed] = None):
    del seed  # Unused. Seed passed to the class.
    outer_dims = self._outer_dims
    if outer_dims is None:
      if self.time_step_spec.observation:
        outer_dims = nest_utils.get_outer_array_shape(
            time_step.observation, self.time_step_spec.observation)
      else:
        outer_dims = ()

    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)

    if observation_and_action_constraint_splitter is not None:
      _, mask = observation_and_action_constraint_splitter(
          time_step.observation)

      zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
      masked_categorical = masked.MaskedCategorical(zero_logits, mask)
      random_action = tf.cast(
          masked_categorical.sample() + self.action_spec.minimum,
          self.action_spec.dtype)

      # If the action spec says each action should be shaped (1,), add another
      # dimension so the final shape is (B, 1) rather than (B,).
      if len(self.action_spec.shape) == 1:
        random_action = tf.expand_dims(random_action, axis=-1)
    else:
      random_action = array_spec.sample_spec_nest(
          self._action_spec, self._rng, outer_dims=outer_dims)

    info = array_spec.sample_spec_nest(
        self._info_spec, self._rng, outer_dims=outer_dims)

    return policy_step.PolicyStep(random_action, policy_state, info)
