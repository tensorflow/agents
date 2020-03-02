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

# Lint as: python2, python3
r"""Sample tf-agents trainer using a mix of graph and out of graph components.

In this example we keep the agent's network and training ops in graph and use a
simple python replay buffer backed by a collections.deque. The python
environment is used and placeholders for the training op are created to allow us
to easily feed in data to train on.

To run:

```bash
tensorboard --logdir $HOME/tmp/dqn_v1/gym/CartPole-v0/ --port 2223 &

python tf_agents/agents/dqn/examples/v1/oog_train_eval.py \
  --root_dir=$HOME/tmp/dqn_v1/gym/CartPole-v0/ \
  --alsologtostderr
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import batched_py_environment
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import q_network
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 10000,
                     'Total number train/eval iterations to perform.')
FLAGS = flags.FLAGS


def collect_step(env, time_step, py_policy, replay_buffer):
  """Steps the environment and collects experience into the replay buffer."""
  action_step = py_policy.action(time_step)
  next_time_step = env.step(action_step.action)
  if not time_step.is_last():
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
  return next_time_step


def train_eval(
    root_dir,
    env_name='CartPole-v0',
    num_iterations=100000,
    fc_layer_params=(100,),
    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints, summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    log_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  """A simple train and eval for DQN."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  # Note this is a python environment.
  env = batched_py_environment.BatchedPyEnvironment([suite_gym.load(env_name)])
  eval_py_env = suite_gym.load(env_name)

  # Convert specs to BoundedTensorSpec.
  action_spec = tensor_spec.from_spec(env.action_spec())
  observation_spec = tensor_spec.from_spec(env.observation_spec())
  time_step_spec = ts.time_step_spec(observation_spec)

  q_net = q_network.QNetwork(
      tensor_spec.from_spec(env.observation_spec()),
      tensor_spec.from_spec(env.action_spec()),
      fc_layer_params=fc_layer_params)

  # The agent must be in graph.
  global_step = tf.compat.v1.train.get_or_create_global_step()
  agent = dqn_agent.DqnAgent(
      time_step_spec,
      action_spec,
      q_network=q_net,
      epsilon_greedy=epsilon_greedy,
      n_step_update=n_step_update,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
      td_errors_loss_fn=common.element_wise_squared_loss,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=global_step)

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_policy.PyTFPolicy(tf_collect_policy)
  greedy_policy = py_tf_policy.PyTFPolicy(agent.policy)
  random_policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                                  env.action_spec())

  # Python replay buffer.
  replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
      capacity=replay_buffer_capacity,
      data_spec=tensor_spec.to_nest_array_spec(agent.collect_data_spec))

  time_step = env.reset()

  # Initialize the replay buffer with some transitions. We use the random
  # policy to initialize the replay buffer to make sure we get a good
  # distribution of actions.
  for _ in range(initial_collect_steps):
    time_step = collect_step(env, time_step, random_policy, replay_buffer)

  # TODO(b/112041045) Use global_step as counter.
  train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir, agent=agent, global_step=global_step)

  policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'policy'),
      policy=agent.policy,
      global_step=global_step)

  ds = replay_buffer.as_dataset(
      sample_batch_size=batch_size, num_steps=n_step_update + 1)
  ds = ds.prefetch(4)
  itr = tf.compat.v1.data.make_initializable_iterator(ds)

  experience = itr.get_next()

  train_op = common.function(agent.train)(experience)

  with eval_summary_writer.as_default(), \
       tf.compat.v2.summary.record_if(True):
    for eval_metric in eval_metrics:
      eval_metric.tf_summaries(train_step=global_step)

  with tf.compat.v1.Session() as session:
    train_checkpointer.initialize_or_restore(session)
    common.initialize_uninitialized_variables(session)
    session.run(itr.initializer)
    # Copy critic network values to the target critic network.
    session.run(agent.initialize())
    train = session.make_callable(train_op)
    global_step_call = session.make_callable(global_step)
    session.run(train_summary_writer.init())
    session.run(eval_summary_writer.init())

    # Compute initial evaluation metrics.
    global_step_val = global_step_call()
    metric_utils.compute_summaries(
        eval_metrics,
        eval_py_env,
        greedy_policy,
        num_episodes=num_eval_episodes,
        global_step=global_step_val,
        log=True,
        callback=eval_metrics_callback,
    )

    timed_at_step = global_step_val
    collect_time = 0
    train_time = 0
    steps_per_second_ph = tf.compat.v1.placeholder(
        tf.float32, shape=(), name='steps_per_sec_ph')
    steps_per_second_summary = tf.compat.v2.summary.scalar(
        name='global_steps_per_sec', data=steps_per_second_ph,
        step=global_step)

    for _ in range(num_iterations):
      start_time = time.time()
      for _ in range(collect_steps_per_iteration):
        time_step = collect_step(env, time_step, collect_policy, replay_buffer)
      collect_time += time.time() - start_time
      start_time = time.time()
      for _ in range(train_steps_per_iteration):
        loss = train()
      train_time += time.time() - start_time
      global_step_val = global_step_call()
      if global_step_val % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step_val, loss.loss)
        steps_per_sec = (
            (global_step_val - timed_at_step) / (collect_time + train_time))
        session.run(
            steps_per_second_summary,
            feed_dict={steps_per_second_ph: steps_per_sec})
        logging.info('%.3f steps/sec', steps_per_sec)
        logging.info('%s', 'collect_time = {}, train_time = {}'.format(
            collect_time, train_time))
        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

      if global_step_val % eval_interval == 0:
        metric_utils.compute_summaries(
            eval_metrics,
            eval_py_env,
            greedy_policy,
            num_episodes=num_eval_episodes,
            global_step=global_step_val,
            log=True,
            callback=eval_metrics_callback,
        )
        # Reset timing to avoid counting eval time.
        timed_at_step = global_step_val
        start_time = time.time()


def main(_):
  tf.compat.v1.enable_resource_variables()
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_resource_variables()
  train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
