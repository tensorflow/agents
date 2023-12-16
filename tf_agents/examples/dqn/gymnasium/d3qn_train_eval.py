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

r"""Example training Double Dueling DQN (D3QN) using actor/learner in a gymnasium environment.

To run D3QN on LunarLander:

```bash
tensorboard --logdir $HOME/tmp/d3qn_lunar_lander --port 2223 &
python tf_agents/examples/dqn/gymnasium/d3qn_train_eval.py
--root_dir=$HOME/tmp/d3qn_lunar_lander
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
from tf_agents.environments import suite_gymnasium as suite_gym
from tf_agents.metrics import py_metrics
from tf_agents.networks import dueling_q_network
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

flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
flags.DEFINE_integer(
    'reverb_port',
    None,
    'Port for reverb server, if None, use a randomly chosen unused port.',
)
flags.DEFINE_integer(
    'num_iterations', 200000, 'Total number train/eval iterations to perform.'
)
flags.DEFINE_integer(
    'eval_interval',
    1000,
    'Number of train steps between evaluations. Set to 0 to skip.',
)
flags.DEFINE_boolean('dueling', False, 'Set to True for dueling DQN')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')


@gin.configurable
def train_eval(
    root_dir,
    env_name='LunarLander-v2',
    # Training params
    initial_collect_steps=1000,
    num_iterations=200000,
    fc_layer_params=(128, 128),
    # Agent params
    epsilon_greedy=0.1,
    min_epsilon=0.0001,
    num_decay_steps=40000,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    target_update_tau=1.0,
    target_update_period=120,
    reward_scale_factor=1.0,
    # Replay params
    reverb_port=None,
    replay_capacity=100000,
    # Others
    policy_save_interval=1000,
    eval_interval=1000,
    eval_episodes=10,
):
  """Trains and evaluates D3QN."""
  collect_env = suite_gym.load(env_name, max_episode_steps=400)
  eval_env = suite_gym.load(
      env_name,
      max_episode_steps=400,
      render_kwargs={'render_mode': 'rgb_array'},
  )

  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env)
  )

  train_step = train_utils.create_train_step()

  # D3QN requires Dueling Q Networks
  q_net = dueling_q_network.DuelingQNetwork(
      input_tensor_spec=observation_tensor_spec,
      action_spec=action_tensor_spec,
      fc_layer_params=fc_layer_params,
  )

  target_q_net = dueling_q_network.DuelingQNetwork(
      input_tensor_spec=observation_tensor_spec,
      action_spec=action_tensor_spec,
      fc_layer_params=fc_layer_params,
  )

  epsilon_decay = tf.compat.v1.train.polynomial_decay(
      learning_rate=epsilon_greedy,
      global_step=train_step,
      decay_steps=num_decay_steps,
      end_learning_rate=min_epsilon,
      power=0.5,
  )

  # D3QN requires a Dueling Double DQN agent
  agent = dqn_agent.D3qnAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      q_network=q_net,
      epsilon_greedy=epsilon_decay,
      n_step_update=n_step_update,
      target_q_network=target_q_net,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      td_errors_loss_fn=common.element_wise_squared_loss,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      train_step_counter=train_step,
  )

  table_name = 'uniform_table'
  sequence_length = n_step_update + 1
  table = reverb.Table(
      table_name,
      max_size=replay_capacity,
      sampler=reverb.selectors.Uniform(),
      # sampler=reverb.selectors.Prioritized(priority_exponent=0.5),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1),
  )
  reverb_server = reverb.Server([table], port=reverb_port)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=sequence_length,
      table_name=table_name,
      local_server=reverb_server,
  )
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client,
      table_name,
      sequence_length=sequence_length,
      stride_length=1,
  )

  dataset = reverb_replay.as_dataset(
      num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
  ).prefetch(3)
  experience_dataset_fn = lambda: dataset

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  env_step_metric = py_metrics.EnvironmentSteps()

  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric},
      ),
      triggers.StepPerSecondLogTrigger(train_step, interval=100),
  ]

  dqn_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
  )

  # If we haven't trained yet make sure we collect some random samples first to
  # fill up the Replay Buffer with some experience.
  random_policy = random_py_policy.RandomPyPolicy(
      collect_env.time_step_spec(), collect_env.action_spec()
  )
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer],
  )
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True
  )

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      observers=[rb_observer, env_step_metric],
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
  )

  tf_greedy_policy = agent.policy
  greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_greedy_policy, use_tf_function=True
  )

  eval_actor = actor.Actor(
      eval_env,
      greedy_policy,
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
    dqn_learner.run(iterations=1)

    if eval_interval and dqn_learner.train_step_numpy % eval_interval == 0:
      logging.info('Evaluating (epsilon: %s).', epsilon_decay())
      eval_actor.run_and_log()
      average_return = eval_actor.metrics[0].result()
      # LunarLander-v2 goal is 200.0
      if average_return > 200.0:
        break

  rb_observer.close()
  reverb_server.stop()


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  train_eval(
      root_dir=FLAGS.root_dir,
      num_iterations=FLAGS.num_iterations,
      reverb_port=FLAGS.reverb_port,
      eval_interval=FLAGS.eval_interval,
  )


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  multiprocessing.handle_main(functools.partial(app.run, main))
