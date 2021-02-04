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
r"""Example training DQN using actor/learner in an Atari environment.

To run DQN on Pong following the hyperparameters from
"Human level control through deep reinforcement learning" (Mnih et al., 2015)
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

```bash
tensorboard --logdir $HOME/tmp/dqn/gym/Pong-v0/ --port 2223 &
python tf_agents/experimental/examples/dqn/mnih15/dqn_train_eval_atari.py \
  --root_dir=/tmp/dqn_pong
```

"""
import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
import reverb
import tensorflow.compat.v2 as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_atari
from tf_agents.metrics import py_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_integer(
    'num_iterations', 50000000,
    'Total number iterations to perform, defaults to 50M collect steps.')
flags.DEFINE_integer(
    'eval_interval', 250000,
    'Number of train steps between evaluations. Set to 0 to skip.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')


def create_q_network(num_actions):
  """Create a Q network following the architecture from Minh 15."""

  kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      activation=tf.keras.activations.relu,
      kernel_initializer=kernel_initializer)
  dense = functools.partial(
      tf.keras.layers.Dense,
      activation=tf.keras.activations.relu,
      kernel_initializer=kernel_initializer)
  logits = functools.partial(
      tf.keras.layers.Dense,
      activation=None,
      kernel_initializer=kernel_initializer)

  return sequential.Sequential(
      # We divide the grayscale pixel values by 255 here rather than storing
      # normalized values because uint8s are 4x cheaper to store than float32s.
      [
          tf.keras.layers.Lambda(lambda x: x / 255),
          conv2d(32, (8, 8), 4),
          conv2d(64, (4, 4), 2),
          conv2d(64, (3, 3), 1),
          tf.keras.layers.Flatten(),
          dense(512),
          logits(num_actions)
      ])


@gin.configurable
def train_eval(
    root_dir,
    env_name='Pong-v0',
    # Training params
    update_frequency=4,  # Number of collect steps per policy update
    initial_collect_steps=50000,  # 50k collect steps
    num_iterations=50000000,  # 50M collect steps
    # Taken from Rainbow as it's not specified in Mnih,15.
    max_episode_frames_collect=50000,  # env frames observed by the agent
    max_episode_frames_eval=108000,  # env frames observed by the agent
    # Agent params
    epsilon_greedy=0.1,
    epsilon_decay_period=250000,  # 1M collect steps / update_frequency
    batch_size=32,
    learning_rate=0.00025,
    n_step_update=1,
    gamma=0.99,
    target_update_tau=1.0,
    target_update_period=2500,  # 10k collect steps / update_frequency
    reward_scale_factor=1.0,
    # Replay params
    reverb_port=None,
    replay_capacity=1000000,
    # Others
    policy_save_interval=250000,
    eval_interval=1000,
    eval_episodes=30,
    debug_summaries=True):
  """Trains and evaluates DQN."""

  collect_env = suite_atari.load(
      env_name,
      max_episode_steps=max_episode_frames_collect,
      gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
  eval_env = suite_atari.load(
      env_name,
      max_episode_steps=max_episode_frames_eval,
      gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)

  unused_observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  train_step = train_utils.create_train_step()

  num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
  epsilon = tf.compat.v1.train.polynomial_decay(
      1.0,
      train_step,
      epsilon_decay_period,
      end_learning_rate=epsilon_greedy)
  agent = dqn_agent.DqnAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      q_network=create_q_network(num_actions),
      epsilon_greedy=epsilon,
      n_step_update=n_step_update,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      optimizer=tf.keras.optimizers.RMSprop(
          learning_rate=learning_rate,
          rho=0.95,
          momentum=0.95,
          epsilon=0.01,
          centered=True),
      td_errors_loss_fn=common.element_wise_huber_loss,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      train_step_counter=train_step,
      debug_summaries=debug_summaries)

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
      reverb_replay.py_client, table_name,
      sequence_length=2,
      stride_length=1)

  dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(3)
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
      triggers.StepPerSecondLogTrigger(train_step, interval=100),
  ]

  dqn_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers)

  # If we haven't trained yet make sure we collect some random samples first to
  # fill up the Replay Buffer with some experience.
  random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(),
                                                  collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy,
                                                      use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=update_frequency,
      observers=[rb_observer, env_step_metric],
      metrics=actor.collect_metrics(10),
      reference_metrics=[env_step_metric],
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
  )

  tf_greedy_policy = agent.policy
  greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_greedy_policy,
                                                     use_tf_function=True)

  eval_actor = actor.Actor(
      eval_env,
      greedy_policy,
      train_step,
      episodes_per_run=eval_episodes,
      metrics=actor.eval_metrics(eval_episodes),
      reference_metrics=[env_step_metric],
      summary_dir=os.path.join(root_dir, 'eval'),
  )

  if eval_interval:
    logging.info('Evaluating.')
    eval_actor.run_and_log()

  logging.info('Training.')
  for _ in range(num_iterations):
    collect_actor.run()
    dqn_learner.run(iterations=1)

    if eval_interval and dqn_learner.train_step_numpy % eval_interval == 0:
      logging.info('Evaluating.')
      eval_actor.run_and_log()

  rb_observer.close()
  reverb_server.stop()


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  train_eval(
      FLAGS.root_dir,
      num_iterations=FLAGS.num_iterations,
      reverb_port=FLAGS.reverb_port,
      eval_interval=FLAGS.eval_interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  multiprocessing.handle_main(functools.partial(app.run, main))
