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

r"""Sample training with distributed collection using a variable container.

See README for launch instructions.
"""

import os
from typing import Callable, Text

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow.compat.v2 as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', None, 'Name of the environment')
flags.DEFINE_string('replay_buffer_server_address', None,
                    'Replay buffer server address.')
flags.DEFINE_string('variable_container_server_address', None,
                    'Variable container server address.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


def _create_agent(train_step: tf.Variable,
                  observation_tensor_spec: types.NestedTensorSpec,
                  action_tensor_spec: types.NestedTensorSpec,
                  time_step_tensor_spec: ts.TimeStep,
                  learning_rate: float) -> tf_agent.TFAgent:
  """Creates an agent."""
  critic_net = critic_network.CriticNetwork(
      (observation_tensor_spec, action_tensor_spec),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=(256, 256),
      kernel_initializer='glorot_uniform',
      last_kernel_initializer='glorot_uniform')

  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_tensor_spec,
      action_tensor_spec,
      fc_layer_params=(256, 256),
      continuous_projection_net=tanh_normal_projection_network
      .TanhNormalProjectionNetwork)

  return sac_agent.SacAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      critic_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=0.99,
      reward_scale_factor=0.1,
      gradient_clipping=None,
      train_step_counter=train_step)


@gin.configurable
def train(
    root_dir: Text,
    environment_name: Text,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    suite_load_fn: Callable[[Text],
                            py_environment.PyEnvironment] = suite_mujoco.load,
    # Training params
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    num_iterations: int = 2000000,
    learner_iterations_per_call: int = 1) -> None:
  """Trains a SAC agent."""
  # Get the specs from the environment.
  logging.info('Training SAC with learning rate: %f', learning_rate)
  env = suite_load_fn(environment_name)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env))

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = _create_agent(
        train_step=train_step,
        observation_tensor_spec=observation_tensor_spec,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        learning_rate=learning_rate)

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weigths.
  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir, agent, train_step, interval=1000)

  # Create the variable container.
  variables = {
      reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.push(variables)

  # Create the replay buffer.
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      server_address=replay_buffer_server_address)

  # Initialize the dataset.
  def experience_dataset_fn():
    with strategy.scope():
      return reverb_replay.as_dataset(
          sample_batch_size=batch_size, num_steps=2).prefetch(3)

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=1000)
  ]
  sac_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
      strategy=strategy)

  # Run the training loop.
  while train_step.numpy() < num_iterations:
    sac_learner.run(iterations=learner_iterations_per_call)
    variable_container.push(variables)


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

  train(
      root_dir=FLAGS.root_dir,
      environment_name=FLAGS.env_name,
      strategy=strategy,
      replay_buffer_server_address=FLAGS.replay_buffer_server_address,
      variable_container_server_address=FLAGS.variable_container_server_address)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir', 'env_name', 'replay_buffer_server_address',
      'variable_container_server_address'
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
