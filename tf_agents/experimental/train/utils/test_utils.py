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

# Lint as: python3
"""Utils for running distributed actor/learner tests."""

import functools

from absl import logging
import numpy as np
import reverb
import tensorflow.compat.v2 as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import suite_gym
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.experimental.train import actor
from tf_agents.experimental.train.utils import replay_buffer_utils
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import sequential
from tf_agents.networks import value_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


def configure_logical_cpus():
  """Configures exactly 4 logical CPUs for the first physical CPU.

  Assumes no logical configuration exists or it was configured the same way.

  **Note**: The reason why the number of logical CPUs fixed is because
  reconfiguring the number of logical CPUs once the underlying runtime has been
  initialized is not supported (raises `RuntimeError`). So, with this choice it
  is ensured that tests running in the same process calling this function
  multiple times do not break.
  """
  first_cpu = tf.config.list_physical_devices('CPU')[0]
  try:
    logical_devices = [
        tf.config.experimental.VirtualDeviceConfiguration() for _ in range(4)
    ]
    tf.config.experimental.set_virtual_device_configuration(
        first_cpu, logical_devices=logical_devices)
    logging.info(
        'No current virtual device configuration. Defining 4 virtual CPUs on '
        'the first physical one.')
  except RuntimeError:
    current_config = tf.config.experimental.get_virtual_device_configuration(
        first_cpu)
    logging.warn(
        'The following virtual device configuration already exists: %s which '
        'resulted this call to fail with `RuntimeError` since it is not '
        'possible to reconfigure it after runtime initialization. It is '
        'probably safe to ignore.', current_config)


def get_cartpole_env_and_specs():
  env = suite_gym.load('CartPole-v0')

  _, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env))

  return env, action_tensor_spec, time_step_tensor_spec


def build_dummy_sequential_net(fc_layer_params, action_spec):
  """Build a dummy sequential network."""
  num_actions = action_spec.maximum - action_spec.minimum + 1

  logits = functools.partial(
      tf.keras.layers.Dense,
      activation=None,
      kernel_initializer=tf.compat.v1.initializers.random_uniform(
          minval=-0.03, maxval=0.03),
      bias_initializer=tf.compat.v1.initializers.constant(-0.2))

  dense = functools.partial(
      tf.keras.layers.Dense,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.compat.v1.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

  return sequential.Sequential(
      [dense(num_units) for num_units in fc_layer_params]
      + [logits(num_actions)])


def create_ppo_agent_and_dataset_fn(action_spec, time_step_spec, train_step,
                                    batch_size):
  """Builds and returns a dummy PPO Agent, dataset and dataset function."""
  del action_spec  # Unused.
  del time_step_spec  # Unused.
  del batch_size  # Unused.

  # No arbitrary spec supported.
  obs_spec = tensor_spec.TensorSpec([2], tf.float32)
  ts_spec = ts.time_step_spec(obs_spec)
  act_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      obs_spec,
      act_spec,
      fc_layer_params=(100,),
      activation_fn=tf.keras.activations.tanh)

  value_net = value_network.ValueNetwork(
      obs_spec, fc_layer_params=(100,), activation_fn=tf.keras.activations.tanh)

  agent = ppo_clip_agent.PPOClipAgent(
      ts_spec,
      act_spec,
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      actor_net=actor_net,
      value_net=value_net,
      entropy_regularization=0.0,
      importance_ratio_clipping=0.2,
      normalize_observations=False,
      normalize_rewards=False,
      use_gae=False,
      use_td_lambda_return=False,
      num_epochs=1,
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=train_step,
      compute_value_and_advantage_in_train=False)

  def _create_experience(_):
    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)
    mid_time_step_val = ts.StepType.MID.tolist()
    time_steps = ts.TimeStep(
        step_type=tf.constant([[mid_time_step_val] * 3] * 2, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    action_distribution_parameters = {
        'loc': tf.constant([[[0.0]] * 3] * 2, dtype=tf.float32),
        'scale': tf.constant([[[1.0]] * 3] * 2, dtype=tf.float32),
    }
    value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                              dtype=tf.float32)

    policy_info = {
        'dist_params': action_distribution_parameters,
    }
    policy_info['value_prediction'] = value_preds
    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)
    return agent._preprocess(experience)  # pylint: disable=protected-access

  dataset = tf.data.Dataset.from_tensor_slices([[i] for i in range(100)
                                               ]).map(_create_experience)
  dataset = tf.data.Dataset.zip((dataset, tf.data.experimental.Counter()))
  dataset_fn = lambda: dataset

  return agent, dataset, dataset_fn, agent.training_data_spec


def create_dqn_agent_and_dataset_fn(action_spec, time_step_spec, train_step,
                                    batch_size):
  """Builds and returns a dataset function for DQN Agent."""
  q_net = build_dummy_sequential_net(fc_layer_params=(100,),
                                     action_spec=action_spec)

  agent = dqn_agent.DqnAgent(
      time_step_spec,
      action_spec,
      q_network=q_net,
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      train_step_counter=train_step)
  agent.initialize()

  def make_item(_):
    traj = tensor_spec.sample_spec_nest(
        agent.collect_data_spec, seed=123, outer_dims=[2])

    def scale_observation_only(item):
      # Scale float values in the sampled item by large value to avoid NaNs.
      if item.dtype == tf.float32:
        return tf.math.divide(item, 1.e+22)
      else:
        return item

    return tf.nest.map_structure(scale_observation_only, traj)

  l = []
  for i in range(100):
    l.append([i])
  dataset = tf.data.Dataset.zip(
      (tf.data.Dataset.from_tensor_slices(l).map(make_item),
       tf.data.experimental.Counter()))
  dataset_fn = lambda: dataset.batch(batch_size)

  return agent, dataset, dataset_fn, agent.collect_data_spec


def build_actor(root_dir, env, agent, rb_observer, train_step):
  """Builds the Actor."""
  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)
  temp_dir = root_dir + 'actor'
  test_actor = actor.Actor(
      env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor.collect_metrics(10),
      summary_dir=temp_dir,
      observers=[rb_observer])

  return test_actor


def get_actor_thread(test_case, reverb_server_port, num_iterations=10):
  """Returns a thread that runs an Actor."""

  def build_and_run_actor():
    root_dir = test_case.create_tempdir().full_path
    env, action_tensor_spec, time_step_tensor_spec = (
        get_cartpole_env_and_specs())

    train_step = train_utils.create_train_step()

    q_net = build_dummy_sequential_net(fc_layer_params=(100,),
                                       action_spec=action_tensor_spec)

    agent = dqn_agent.DqnAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        q_network=q_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        train_step_counter=train_step)

    _, rb_observer = (
        replay_buffer_utils.get_reverb_buffer_and_observer(
            agent.collect_data_spec,
            table_name=reverb_replay_buffer.DEFAULT_TABLE,
            sequence_length=2,
            reverb_server_address='localhost:{}'.format(reverb_server_port)))

    variable_container = reverb_variable_container.ReverbVariableContainer(
        server_address='localhost:{}'.format(reverb_server_port),
        table_names=[reverb_variable_container.DEFAULT_TABLE])

    test_actor = build_actor(
        root_dir, env, agent, rb_observer, train_step)

    variables_dict = {
        reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
        reverb_variable_container.TRAIN_STEP_KEY: train_step
    }
    variable_container.update(variables_dict)

    for _ in range(num_iterations):
      test_actor.run()

  actor_thread = test_case.checkedThread(target=build_and_run_actor)
  return actor_thread


def check_variables_different(test_case, old_vars_numpy, new_vars_numpy):
  """Tests whether the two sets of variables are different.

  Useful for checking if variables were updated, i.e. a train step was run.

  Args:
    test_case: an instande of tf.test.TestCase for assertions
    old_vars_numpy: numpy representation of old variables
    new_vars_numpy: numpy representation of new variables
  """

  # Check if there is a change.
  def changed(a, b):
    return not np.equal(a, b).all()

  vars_changed = tf.nest.flatten(
      tf.nest.map_structure(changed, old_vars_numpy, new_vars_numpy))

  # Assert if any of the variable changed.
  test_case.assertTrue(np.any(vars_changed))


def create_reverb_server_for_replay_buffer_and_variable_container(
    collect_policy, train_step, replay_buffer_capacity, port):
  """Sets up one reverb server for replay buffer and variable container."""
  # Create the signature for the variable container holding the policy weights.
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
      variables)

  # Create the signature for the replay buffer holding observed experience.
  replay_buffer_signature = tensor_spec.from_spec(
      collect_policy.collect_data_spec)

  # Crete and start the replay buffer and variable container server.
  server = reverb.Server(
      tables=[
          reverb.Table(  # Replay buffer storing experience.
              name=reverb_replay_buffer.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              # TODO(b/159073060): Set rate limiter for SAC properly.
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_buffer_capacity,
              max_times_sampled=0,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=1,
              max_times_sampled=0,
              signature=variable_container_signature,
          ),
      ],
      port=port)
  return server
