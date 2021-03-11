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

"""Utility classes and functions for unit tests for PPOLearner."""

import numpy as np
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


# We could not use a mock directly because the Learner requires the agent to
# be `Trackable`.
class FakePPOAgent(ppo_agent.PPOAgent):
  """A fake PPO agent that tracks input trajectories into agent.train."""

  def __init__(self, strategy=None):

    self._strategy = strategy

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
        tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1),
        fc_layer_params=(1,),
        activation_fn=tf.nn.tanh)
    value_net = value_network.ValueNetwork(
        tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
        fc_layer_params=(1,))

    super(FakePPOAgent, self).__init__(
        time_step_spec=ts.time_step_spec(
            tensor_spec.TensorSpec(shape=[], dtype=tf.float32)),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1),
        actor_net=actor_net,
        value_net=value_net,
        # Ensures value_prediction, return and normalized_advantage are included
        # as part of the training_data_spec.
        compute_value_and_advantage_in_train=False,
        update_normalizers_in_train=False,
    )
    self.train_called_times = tf.Variable(0, dtype=tf.int32)
    self.experiences = []

  def _train(self, experience, weights):
    self.train_called_times.assign_add(1)
    print('append experience')
    print(experience)
    self.experiences.append(experience)
    if self._strategy is None:
      return tf_agent.LossInfo(0., 0.)
    else:
      batch_zero = tf.constant([0.] * 10, dtype=tf.float32)
      return tf_agent.LossInfo(batch_zero, batch_zero)

  def reset(self):
    self.train_called_times.assign(0)
    self.experiences = []


def create_trajectories(n_time_steps, batch_size):
  """Create an input trajectory of shape [batch_size, n_time_steps, ...]."""

  # Observation looks like:
  # [[ 0.,  1., ... n_time_steps.],
  #  [10., 11., ... n_time_steps.],
  #  [20., 21., ... n_time_steps.],
  #  [ ...                       ],
  #  [10*batch_size., ... 10*batch_size+n_time_steps.]]
  observation_array = np.asarray(
      [np.arange(n_time_steps) + 10 * i for i in range(batch_size)])
  observations = tf.convert_to_tensor(observation_array, dtype=tf.float32)

  default_tensor = tf.constant(
      [[1] * n_time_steps] * batch_size, dtype=tf.float32)
  mid_time_step_val = ts.StepType.MID.tolist()
  time_steps = ts.TimeStep(
      step_type=tf.constant(
          [[mid_time_step_val] * n_time_steps] * batch_size, dtype=tf.int32),
      reward=default_tensor,
      discount=default_tensor,
      observation=observations)
  actions = tf.constant([[[1]] * n_time_steps] * batch_size, dtype=tf.float32)
  policy_info = {
      'dist_params': {
          'loc':
              tf.constant(
                  [[[1]] * n_time_steps] * batch_size, dtype=tf.float32),
          'scale':
              tf.constant(
                  [[[1]] * n_time_steps] * batch_size, dtype=tf.float32)
      },
      'value_prediction': default_tensor,
      'return': default_tensor,
      'advantage': default_tensor,
  }
  return trajectory.Trajectory(time_steps.step_type, observations, actions,
                               policy_info, time_steps.step_type,
                               time_steps.reward, time_steps.discount)
