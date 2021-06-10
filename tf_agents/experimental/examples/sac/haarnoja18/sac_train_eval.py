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

# Lint as: python3
r"""Train and Eval SAC.

All hyperparameters come from the SAC paper
https://arxiv.org/pdf/1812.05905.pdf
"""
import functools
import os

from absl import app
from absl import flags
from absl import logging

import gin
import reverb
import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_mujoco
from tf_agents.keras_layers import inner_reshape
from tf_agents.metrics import py_metrics
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_integer('num_iterations', 3000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer(
    'eval_interval', 10000,
    'Number of train steps between evaluations. Set to 0 to skip.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')


dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer='glorot_uniform')


def create_fc_network(layer_units):
  return sequential.Sequential([dense(num_units) for num_units in layer_units])


def create_identity_layer():
  return tf.keras.layers.Lambda(lambda x: x)


def create_sequential_critic_network(obs_fc_layer_units,
                                     action_fc_layer_units,
                                     joint_fc_layer_units):
  """Create a sequential critic network."""
  # Split the inputs into observations and actions.
  def split_inputs(inputs):
    return {'observation': inputs[0], 'action': inputs[1]}

  # Create an observation network.
  obs_network = (create_fc_network(obs_fc_layer_units) if obs_fc_layer_units
                 else create_identity_layer())

  # Create an action network.
  action_network = (create_fc_network(action_fc_layer_units)
                    if action_fc_layer_units else create_identity_layer())

  # Create a joint network.
  joint_network = (create_fc_network(joint_fc_layer_units)
                   if joint_fc_layer_units else create_identity_layer())

  # Final layer.
  value_layer = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

  return sequential.Sequential([
      tf.keras.layers.Lambda(split_inputs),
      nest_map.NestMap({
          'observation': obs_network,
          'action': action_network
      }),
      nest_map.NestFlatten(),
      tf.keras.layers.Concatenate(),
      joint_network,
      value_layer,
      inner_reshape.InnerReshape(current_shape=[1], new_shape=[])
  ], name='sequential_critic')


class _TanhNormalProjectionNetworkWrapper(
    tanh_normal_projection_network.TanhNormalProjectionNetwork):
  """Wrapper to pass predefined `outer_rank` to underlying projection net."""

  def __init__(self, sample_spec, predefined_outer_rank=1):
    super(_TanhNormalProjectionNetworkWrapper, self).__init__(sample_spec)
    self.predefined_outer_rank = predefined_outer_rank

  def call(self, inputs, network_state=(), **kwargs):
    kwargs['outer_rank'] = self.predefined_outer_rank
    if 'step_type' in kwargs:
      del kwargs['step_type']
    return super(_TanhNormalProjectionNetworkWrapper,
                 self).call(inputs, **kwargs)


def create_sequential_actor_network(actor_fc_layers, action_tensor_spec):
  """Create a sequential actor network."""
  def tile_as_nest(non_nested_output):
    return tf.nest.map_structure(lambda _: non_nested_output,
                                 action_tensor_spec)

  return sequential.Sequential(
      [dense(num_units) for num_units in actor_fc_layers] +
      [tf.keras.layers.Lambda(tile_as_nest)] + [
          nest_map.NestMap(
              tf.nest.map_structure(_TanhNormalProjectionNetworkWrapper,
                                    action_tensor_spec))
      ])


@gin.configurable
def train_eval(
    root_dir,
    strategy: tf.distribute.Strategy,
    env_name='HalfCheetah-v2',
    # Training params
    initial_collect_steps=10000,
    num_iterations=3200000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Agent params
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    gamma=0.99,
    target_update_tau=0.005,
    target_update_period=1,
    reward_scale_factor=0.1,
    # Replay params
    reverb_port=None,
    replay_capacity=1000000,
    # Others
    policy_save_interval=10000,
    replay_buffer_save_interval=100000,
    eval_interval=10000,
    eval_episodes=30,
    debug_summaries=False,
    summarize_grads_and_vars=False):
  """Trains and evaluates SAC."""
  logging.info('Training SAC on: %s', env_name)
  collect_env = suite_mujoco.load(env_name)
  eval_env = suite_mujoco.load(env_name)

  _, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  actor_net = create_sequential_actor_network(
      actor_fc_layers=actor_fc_layers, action_tensor_spec=action_tensor_spec)

  critic_net = create_sequential_critic_network(
      obs_fc_layer_units=critic_obs_fc_layers,
      action_fc_layer_units=critic_action_fc_layers,
      joint_fc_layer_units=critic_joint_fc_layers)

  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = sac_agent.SacAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step)
    agent.initialize()

  table_name = 'uniform_table'
  table = reverb.Table(
      table_name,
      max_size=replay_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1))

  reverb_checkpoint_dir = os.path.join(root_dir, learner.TRAIN_DIR,
                                       learner.REPLAY_BUFFER_CHECKPOINT_DIR)
  reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
      path=reverb_checkpoint_dir)
  reverb_server = reverb.Server([table],
                                port=reverb_port,
                                checkpointer=reverb_checkpointer)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name=table_name,
      local_server=reverb_server)
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client,
      table_name,
      sequence_length=2,
      stride_length=1)

  dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
  experience_dataset_fn = lambda: dataset

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  env_step_metric = py_metrics.EnvironmentSteps()
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric}),
      triggers.ReverbCheckpointTrigger(
          train_step,
          interval=replay_buffer_save_interval,
          reverb_client=reverb_replay.py_client),
      # TODO(b/165023684): Add SIGTERM handler to checkpoint before preemption.
      triggers.StepPerSecondLogTrigger(train_step, interval=1000),
  ]

  agent_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
      strategy=strategy)

  random_policy = random_py_policy.RandomPyPolicy(
      collect_env.time_step_spec(), collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
      observers=[rb_observer, env_step_metric])

  tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_greedy_policy, use_tf_function=True)

  eval_actor = actor.Actor(
      eval_env,
      eval_greedy_policy,
      train_step,
      episodes_per_run=eval_episodes,
      metrics=actor.eval_metrics(eval_episodes),
      summary_dir=os.path.join(root_dir, 'eval'),
  )

  if eval_interval:
    logging.info('Evaluating.')
    eval_actor.run_and_log()

  logging.info('Training.')
  for _ in range(num_iterations):
    collect_actor.run()
    agent_learner.run(iterations=1)

    if eval_interval and agent_learner.train_step_numpy % eval_interval == 0:
      logging.info('Evaluating.')
      eval_actor.run_and_log()

  rb_observer.close()
  reverb_server.stop()


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

  train_eval(
      FLAGS.root_dir,
      strategy=strategy,
      num_iterations=FLAGS.num_iterations,
      reverb_port=FLAGS.reverb_port,
      eval_interval=FLAGS.eval_interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
