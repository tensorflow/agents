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

# Lint as: python2, python3
"""Tests for tf_agents.agents.ppo.ppo_actor_network."""

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ppo import ppo_policy
from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class DeterministicSeedStream(object):
  """A fake seed stream class that always generates a deterministic seed."""

  def __init__(self, seed, salt=''):
    del salt
    self._seed = seed

  def __call__(self):
    return self._seed


class PPOAgentActorDist(ppo_agent.PPOAgent):

  def __init__(self):

    observation_tensor_spec = tf.TensorSpec(shape=[1], dtype=tf.float32)
    action_tensor_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(1,),
        activation_fn=tf.nn.tanh,
        kernel_initializer=tf.keras.initializers.Orthogonal(seed=1),
        seed_stream_class=DeterministicSeedStream,
        seed=1)

    value_net = value_network.ValueNetwork(
        observation_tensor_spec, fc_layer_params=(1,))

    super(PPOAgentActorDist, self).__init__(
        time_step_spec=ts.time_step_spec(observation_tensor_spec),
        action_spec=action_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        # Ensures value_prediction, return and advantage are included as parts
        # of the training_data_spec.
        compute_value_and_advantage_in_train=True,
        update_normalizers_in_train=False,
        optimizer=tf.compat.v1.train.AdamOptimizer(),
    )
    # There is an artifical call on `_train` during the initialization which
    # ensures that the variables of the optimizer are initialized. This is
    # excluded from the call count.
    self.train_called_times = -1
    self.experiences = []


class PPOAgentSequential(ppo_agent.PPOAgent):

  def __init__(self):

    observation_tensor_spec = tf.TensorSpec(shape=[1], dtype=tf.float32)
    action_tensor_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

    actor_net_lib = ppo_actor_network.PPOActorNetwork()
    actor_net_lib.seed_stream_class = DeterministicSeedStream
    actor_net = actor_net_lib.create_sequential_actor_net(
        fc_layer_units=(1,), action_tensor_spec=action_tensor_spec, seed=1)
    value_net = value_network.ValueNetwork(
        observation_tensor_spec, fc_layer_params=(1,))

    super(PPOAgentSequential, self).__init__(
        time_step_spec=ts.time_step_spec(observation_tensor_spec),
        action_spec=action_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        # Ensures value_prediction, return and advantage are included as parts
        # of the training_data_spec.
        compute_value_and_advantage_in_train=True,
        update_normalizers_in_train=False,
    )
    # There is an artifical call on `_train` during the initialization which
    # ensures that the variables of the optimizer are initialized. This is
    # excluded from the call count.
    self.train_called_times = -1
    self.experiences = []


def _create_trajectories(n_time_steps, batch_size, policy_info_scale_name):
  # Observation looks like:
  # [[ 0.,  1., ... n_time_steps.],
  #  [10., 11., ... n_time_steps.],
  #  [20., 21., ... n_time_steps.],
  #  [ ...                       ],
  #  [10*batch_size., ... 10*batch_size+n_time_steps.]]
  observation_array = np.asarray(
      [np.arange(n_time_steps) + 10 * i for i in range(batch_size)])
  # Adding an inner most dimension to fit the observation spec defined above.
  observation_array = np.expand_dims(observation_array, axis=2)
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
          policy_info_scale_name:
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


class PpoActorNetworkTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(PpoActorNetworkTest, self).setUp()
    # Run in full eager mode in order to inspect the content of tensors.
    tf.config.experimental_run_functions_eagerly(True)

  def tearDown(self):
    tf.config.experimental_run_functions_eagerly(False)
    super(PpoActorNetworkTest, self).tearDown()

  def test_same_initialization(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping test: sequential networks not supported in TF1')
    agent_seq = PPOAgentSequential()
    seq_variables = agent_seq._actor_net.trainable_weights

    agent_act_dist = PPOAgentActorDist()
    act_dist_variables = agent_act_dist._actor_net.trainable_weights

    self.assertEqual(len(seq_variables), len(act_dist_variables))
    set_seq_weights = set()
    set_act_dist_weights = set()
    for i in range(len(seq_variables)):
      set_seq_weights.add(seq_variables[i].numpy().sum())
      set_act_dist_weights.add(act_dist_variables[i].numpy().sum())
    self.assertSetEqual(set_seq_weights, set_act_dist_weights)

  def test_no_mismatched_shape(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping test: sequential networks not supported in TF1')
    observation_tensor_spec = tf.TensorSpec(shape=[1], dtype=tf.float32)
    action_tensor_spec = tensor_spec.BoundedTensorSpec((8,), tf.float32, -1, 1)

    actor_net_lib = ppo_actor_network.PPOActorNetwork()
    actor_net_lib.seed_stream_class = DeterministicSeedStream
    actor_net = actor_net_lib.create_sequential_actor_net(
        fc_layer_units=(1,), action_tensor_spec=action_tensor_spec, seed=1)

    actor_output_spec = actor_net.create_variables(observation_tensor_spec)

    distribution_utils.assert_specs_are_compatible(
        actor_output_spec, action_tensor_spec,
        'actor_network output spec does not match action spec')

  def test_same_actor_net_output(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping test: sequential networks not supported in TF1')
    observation_tensor_spec = tf.TensorSpec(shape=[1], dtype=tf.float32)
    action_tensor_spec = tensor_spec.BoundedTensorSpec((8,), tf.float32, -1, 1)

    actor_net_lib = ppo_actor_network.PPOActorNetwork()
    actor_net_lib.seed_stream_class = DeterministicSeedStream
    actor_net_sequential = actor_net_lib.create_sequential_actor_net(
        fc_layer_units=(1,), action_tensor_spec=action_tensor_spec, seed=1)

    actor_net_actor_dist = actor_distribution_network.ActorDistributionNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(1,),
        activation_fn=tf.nn.tanh,
        kernel_initializer=tf.keras.initializers.Orthogonal(
            seed=1),
        seed_stream_class=DeterministicSeedStream,
        seed=1)

    sample_observation = tf.constant([[1], [2]], dtype=tf.float32)
    sequential_output_dist, _ = actor_net_sequential(
        sample_observation, step_type=ts.StepType.MID, network_state=())
    actor_dist_output_dist, _ = actor_net_actor_dist(
        sample_observation, step_type=ts.StepType.MID, network_state=())
    self.assertAllEqual(sequential_output_dist.mean(),
                        actor_dist_output_dist.mean())
    self.assertAllEqual(sequential_output_dist.stddev(),
                        actor_dist_output_dist.stddev())

  def test_same_policy_same_output(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping test: sequential networks not supported in TF1')
    observation_tensor_spec = tf.TensorSpec(shape=[1], dtype=tf.float32)
    action_tensor_spec = tensor_spec.BoundedTensorSpec((8,), tf.float32, -1, 1)

    value_net = value_network.ValueNetwork(
        observation_tensor_spec, fc_layer_params=(1,))

    actor_net_lib = ppo_actor_network.PPOActorNetwork()
    actor_net_lib.seed_stream_class = DeterministicSeedStream
    actor_net_sequential = actor_net_lib.create_sequential_actor_net(
        fc_layer_units=(1,), action_tensor_spec=action_tensor_spec, seed=1)
    actor_net_actor_dist = actor_distribution_network.ActorDistributionNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(1,),
        activation_fn=tf.nn.tanh,
        kernel_initializer=tf.keras.initializers.Orthogonal(seed=1),
        seed_stream_class=DeterministicSeedStream,
        seed=1)

    seq_policy = ppo_policy.PPOPolicy(
        ts.time_step_spec(observation_tensor_spec),
        action_tensor_spec,
        actor_net_sequential,
        value_net,
        collect=True)
    actor_dist_policy = ppo_policy.PPOPolicy(
        ts.time_step_spec(observation_tensor_spec),
        action_tensor_spec,
        actor_net_actor_dist,
        value_net,
        collect=True)

    sample_timestep = ts.TimeStep(
        step_type=tf.constant([1, 1], dtype=tf.int32),
        reward=tf.constant([1, 1], dtype=tf.float32),
        discount=tf.constant([1, 1], dtype=tf.float32),
        observation=tf.constant([[1], [2]], dtype=tf.float32))
    seq_policy_step = seq_policy._distribution(sample_timestep, policy_state=())
    act_dist_policy_step = actor_dist_policy._distribution(
        sample_timestep, policy_state=())

    seq_scale = seq_policy_step.info['dist_params']['scale_diag']
    act_dist_scale = act_dist_policy_step.info['dist_params']['scale']
    self.assertAllEqual(seq_scale, act_dist_scale)
    self.assertAllEqual(seq_policy_step.info['dist_params']['loc'],
                        act_dist_policy_step.info['dist_params']['loc'])

  def test_same_agent_train(self):
    if not tf.executing_eagerly():
      self.skipTest('Skipping test: sequential networks not supported in TF1')

    agent_seq = PPOAgentSequential()
    agent_act_dist = PPOAgentActorDist()
    batch_size = 2
    n_time_steps = 3

    action_distribution_parameters = {
        'loc':
            tf.constant(
                [[[0.0]] * n_time_steps] * batch_size, dtype=tf.float32),
        'scale':
            tf.constant(
                [[[1.0]] * n_time_steps] * batch_size, dtype=tf.float32),
    }
    observations = tf.constant(
        [
            [[1], [3], [5]],
            [[2], [4], [6]],
        ],
        dtype=tf.float32)

    mid_time_step_val = ts.StepType.MID.tolist()
    time_steps = ts.TimeStep(
        step_type=tf.constant(
            [[mid_time_step_val] * n_time_steps] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * n_time_steps] * batch_size, dtype=tf.float32),
        discount=tf.constant(
            [[1] * n_time_steps] * batch_size, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                              dtype=tf.float32)
    policy_info = {
        'dist_params': action_distribution_parameters,
        'value_prediction': value_preds,
    }
    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)
    act_dist_loss = agent_act_dist.train(experience).loss

    policy_info['dist_params']['scale_diag'] = policy_info['dist_params'].pop(
        'scale')
    seq_loss = agent_seq.train(experience).loss
    self.assertAllEqual(act_dist_loss, seq_loss)

    seq_variables = agent_seq._actor_net.trainable_weights
    act_dist_variables = agent_act_dist._actor_net.trainable_weights

    self.assertEqual(len(seq_variables), len(act_dist_variables))
    set_seq_weights = set()
    set_act_dist_weights = set()
    for i in range(len(seq_variables)):
      set_seq_weights.add(seq_variables[i].numpy().sum())
      set_act_dist_weights.add(act_dist_variables[i].numpy().sum())
    self.assertSetEqual(set_seq_weights, set_act_dist_weights)


if __name__ == '__main__':
  tf.test.main()
