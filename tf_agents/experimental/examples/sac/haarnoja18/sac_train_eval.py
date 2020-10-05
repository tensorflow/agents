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
r"""Train and Eval SAC.

All hyperparameters come from the SAC paper
https://arxiv.org/pdf/1812.05905.pdf
"""
import os

from absl import app
from absl import flags
from absl import logging

import gin
import reverb
import tensorflow.compat.v2 as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_mujoco
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

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


@gin.configurable
def train_eval(
    root_dir,
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
    # Defaults to not checkpointing saved policy. If you wish to enable this,
    # please note the caveat explained in README.md.
    policy_save_interval=-1,
    eval_interval=10000,
    eval_episodes=30,
    debug_summaries=False,
    summarize_grads_and_vars=False):
  """Trains and evaluates SAC."""
  logging.info('Training SAC on: %s', env_name)
  collect_env = suite_mujoco.load(env_name)
  eval_env = suite_mujoco.load(env_name)

  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  train_step = train_utils.create_train_step()

  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_tensor_spec,
      action_tensor_spec,
      fc_layer_params=actor_fc_layers,
      continuous_projection_net=tanh_normal_projection_network
      .TanhNormalProjectionNetwork)
  critic_net = critic_network.CriticNetwork(
      (observation_tensor_spec, action_tensor_spec),
      observation_fc_layer_params=critic_obs_fc_layers,
      action_fc_layer_params=critic_action_fc_layers,
      joint_fc_layer_params=critic_joint_fc_layers,
      kernel_initializer='glorot_uniform',
      last_kernel_initializer='glorot_uniform')

  agent = sac_agent.SacAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_learning_rate),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
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

  reverb_server = reverb.Server([table], port=reverb_port)
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
      triggers.StepPerSecondLogTrigger(train_step, interval=1000),
  ]

  agent_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers)

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
  tf.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  train_eval(
      FLAGS.root_dir,
      num_iterations=FLAGS.num_iterations,
      reverb_port=FLAGS.reverb_port,
      eval_interval=FLAGS.eval_interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
