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

r"""Train and Eval DQN on Atari environments.

Training and evaluation proceeds alternately in iterations, where each
iteration consists of a 1M frame training phase followed by a 500K frame
evaluation phase. In the literature, some papers report averages of the train
phases, while others report averages of the eval phases.

This example is configured to use dopamine.atari.preprocessing, which, among
other things, repeats every action it receives for 4 frames, and then returns
the max-pool over the last 2 frames in the group. In this example, when we
refer to "ALE frames" we refer to the frames before the max-pooling step (i.e.
the raw data available for processing). Because of this, many of the
configuration parameters (like initial_collect_steps) are divided by 4 in the
body of the trainer (e.g. if you want to evaluate with 400 frames in the
initial collection, you actually only need to .step the environment 100 times).

For a good survey of training on Atari, see Machado, et al. 2017:
https://arxiv.org/pdf/1709.06009.pdf.

To run:

```bash
tf_agents/agents/dqn/examples/v1/train_eval_atari \
  --root_dir=$HOME/atari/pong \
  --atari_roms_path=/tmp
  --alsologtostderr
```

Additional flags are available such as `--replay_buffer_capacity` and
`--n_step_update`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import batched_py_environment
from tf_agents.environments import suite_atari
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.networks import q_network
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import py_hashed_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import timer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('game_name', 'Pong', 'Name of Atari game to run.')
flags.DEFINE_integer('num_iterations', None,
                     'Number of train/eval iterations to run.')
flags.DEFINE_integer('initial_collect_steps', None,
                     'Number of frames to ALE frames to process before '
                     'beginning to train. Since this is in ALE frames, there '
                     'will be initial_collect_steps/4 items in the replay '
                     'buffer when training starts.')
flags.DEFINE_integer('replay_buffer_capacity', None,
                     'Maximum number of items to store in the replay buffer.')
flags.DEFINE_integer('train_steps_per_iteration', None,
                     'Number of ALE frames to run through for each iteration '
                     'of training.')
flags.DEFINE_integer('n_step_update', None, 'The number of steps to consider '
                     'when computing TD error and TD loss.')
flags.DEFINE_integer('eval_steps_per_iteration', None,
                     'Number of ALE frames to run through for each iteration '
                     'of evaluation.')
FLAGS = flags.FLAGS

# AtariPreprocessing runs 4 frames at a time, max-pooling over the last 2
# frames. We need to account for this when computing things like update
# intervals.
ATARI_FRAME_SKIP = 4


class AtariQNetwork(q_network.QNetwork):
  """QNetwork subclass that divides observations by 255."""

  def call(self, observation, step_type=None, network_state=None):
    state = tf.cast(observation, tf.float32)
    # We divide the grayscale pixel values by 255 here rather than storing
    # normalized values beause uint8s are 4x cheaper to store than float32s.
    state = state / 255
    return super(AtariQNetwork, self).call(
        state, step_type=step_type, network_state=network_state)


def log_metric(metric, prefix):
  tag = common.join_scope(prefix, metric.name)
  logging.info('%s', '{0} = {1}'.format(tag, metric.result()))


@gin.configurable
class TrainEval(object):
  """Train and evaluate DQN on Atari."""

  def __init__(
      self,
      root_dir,
      env_name,
      num_iterations=200,
      max_episode_frames=108000,  # ALE frames
      terminal_on_life_loss=False,
      conv_layer_params=((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)),
      fc_layer_params=(512,),
      # Params for collect
      initial_collect_steps=80000,  # ALE frames
      epsilon_greedy=0.01,
      epsilon_decay_period=1000000,  # ALE frames
      replay_buffer_capacity=1000000,
      # Params for train
      train_steps_per_iteration=1000000,  # ALE frames
      update_period=16,  # ALE frames
      target_update_tau=1.0,
      target_update_period=32000,  # ALE frames
      batch_size=32,
      learning_rate=2.5e-4,
      n_step_update=1,
      gamma=0.99,
      reward_scale_factor=1.0,
      gradient_clipping=None,
      # Params for eval
      do_eval=True,
      eval_steps_per_iteration=500000,  # ALE frames
      eval_epsilon_greedy=0.001,
      # Params for checkpoints, summaries, and logging
      log_interval=1000,
      summary_interval=1000,
      summaries_flush_secs=10,
      debug_summaries=False,
      summarize_grads_and_vars=False,
      eval_metrics_callback=None):
    """A simple Atari train and eval for DQN.

    Args:
      root_dir: Directory to write log files to.
      env_name: Fully-qualified name of the Atari environment (i.e. Pong-v0).
      num_iterations: Number of train/eval iterations to run.
      max_episode_frames: Maximum length of a single episode, in ALE frames.
      terminal_on_life_loss: Whether to simulate an episode termination when a
        life is lost.
      conv_layer_params: Params for convolutional layers of QNetwork.
      fc_layer_params: Params for fully connected layers of QNetwork.
      initial_collect_steps: Number of frames to ALE frames to process before
        beginning to train. Since this is in ALE frames, there will be
        initial_collect_steps/4 items in the replay buffer when training starts.
      epsilon_greedy: Final epsilon value to decay to for training.
      epsilon_decay_period: Period over which to decay epsilon, from 1.0 to
        epsilon_greedy (defined above).
      replay_buffer_capacity: Maximum number of items to store in the replay
        buffer.
      train_steps_per_iteration: Number of ALE frames to run through for each
        iteration of training.
      update_period: Run a train operation every update_period ALE frames.
      target_update_tau: Coeffecient for soft target network updates (1.0 ==
        hard updates).
      target_update_period: Period, in ALE frames, to copy the live network to
        the target network.
      batch_size: Number of frames to include in each training batch.
      learning_rate: RMS optimizer learning rate.
      n_step_update: The number of steps to consider when computing TD error and
        TD loss. Applies standard single-step updates when set to 1.
      gamma: Discount for future rewards.
      reward_scale_factor: Scaling factor for rewards.
      gradient_clipping: Norm length to clip gradients.
      do_eval: If True, run an eval every iteration. If False, skip eval.
      eval_steps_per_iteration: Number of ALE frames to run through for each
        iteration of evaluation.
      eval_epsilon_greedy: Epsilon value to use for the evaluation policy (0 ==
        totally greedy policy).
      log_interval: Log stats to the terminal every log_interval training
        steps.
      summary_interval: Write TF summaries every summary_interval training
        steps.
      summaries_flush_secs: Flush summaries to disk every summaries_flush_secs
        seconds.
      debug_summaries: If True, write additional summaries for debugging (see
        dqn_agent for which summaries are written).
      summarize_grads_and_vars: Include gradients in summaries.
      eval_metrics_callback: A callback function that takes (metric_dict,
        global_step) as parameters. Called after every eval with the results of
        the evaluation.
    """
    self._update_period = update_period / ATARI_FRAME_SKIP
    self._train_steps_per_iteration = (train_steps_per_iteration
                                       / ATARI_FRAME_SKIP)
    self._do_eval = do_eval
    self._eval_steps_per_iteration = eval_steps_per_iteration / ATARI_FRAME_SKIP
    self._eval_epsilon_greedy = eval_epsilon_greedy
    self._initial_collect_steps = initial_collect_steps / ATARI_FRAME_SKIP
    self._summary_interval = summary_interval
    self._num_iterations = num_iterations
    self._log_interval = log_interval
    self._eval_metrics_callback = eval_metrics_callback

    with gin.unlock_config():
      gin.bind_parameter('AtariPreprocessing.terminal_on_life_loss',
                         terminal_on_life_loss)

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()
    self._train_summary_writer = train_summary_writer

    self._eval_summary_writer = None
    if self._do_eval:
      self._eval_summary_writer = tf.compat.v2.summary.create_file_writer(
          eval_dir, flush_millis=summaries_flush_secs * 1000)
      self._eval_metrics = [
          py_metrics.AverageReturnMetric(
              name='PhaseAverageReturn', buffer_size=np.inf),
          py_metrics.AverageEpisodeLengthMetric(
              name='PhaseAverageEpisodeLength', buffer_size=np.inf),
      ]

    self._global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(self._global_step % self._summary_interval, 0)):
      self._env = suite_atari.load(
          env_name,
          max_episode_steps=max_episode_frames / ATARI_FRAME_SKIP,
          gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
      self._env = batched_py_environment.BatchedPyEnvironment([self._env])

      observation_spec = tensor_spec.from_spec(self._env.observation_spec())
      time_step_spec = ts.time_step_spec(observation_spec)
      action_spec = tensor_spec.from_spec(self._env.action_spec())

      with tf.device('/cpu:0'):
        epsilon = tf.compat.v1.train.polynomial_decay(
            1.0,
            self._global_step,
            epsilon_decay_period / ATARI_FRAME_SKIP / self._update_period,
            end_learning_rate=epsilon_greedy)

      with tf.device('/gpu:0'):
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)
        q_net = AtariQNetwork(
            observation_spec,
            action_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params)
        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_network=q_net,
            optimizer=optimizer,
            epsilon_greedy=epsilon,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=(
                target_update_period / ATARI_FRAME_SKIP / self._update_period),
            td_errors_loss_fn=dqn_agent.element_wise_huber_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=self._global_step)

        self._collect_policy = py_tf_policy.PyTFPolicy(
            agent.collect_policy)

        if self._do_eval:
          self._eval_policy = py_tf_policy.PyTFPolicy(
              epsilon_greedy_policy.EpsilonGreedyPolicy(
                  policy=agent.policy,
                  epsilon=self._eval_epsilon_greedy))

        py_observation_spec = self._env.observation_spec()
        py_time_step_spec = ts.time_step_spec(py_observation_spec)
        py_action_spec = policy_step.PolicyStep(self._env.action_spec())
        data_spec = trajectory.from_transition(
            py_time_step_spec, py_action_spec, py_time_step_spec)
        self._replay_buffer = (
            py_hashed_replay_buffer.PyHashedReplayBuffer(
                data_spec=data_spec, capacity=replay_buffer_capacity))

      with tf.device('/cpu:0'):
        ds = self._replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=n_step_update + 1)
        ds = ds.prefetch(4)
        ds = ds.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

      with tf.device('/gpu:0'):
        self._ds_itr = tf.compat.v1.data.make_one_shot_iterator(ds)
        experience = self._ds_itr.get_next()
        self._train_op = agent.train(experience)

        self._env_steps_metric = py_metrics.EnvironmentSteps()
        self._step_metrics = [
            py_metrics.NumberOfEpisodes(),
            self._env_steps_metric,
        ]
        self._train_metrics = self._step_metrics + [
            py_metrics.AverageReturnMetric(buffer_size=10),
            py_metrics.AverageEpisodeLengthMetric(buffer_size=10),
        ]
        # The _train_phase_metrics average over an entire train iteration,
        # rather than the rolling average of the last 10 episodes.
        self._train_phase_metrics = [
            py_metrics.AverageReturnMetric(
                name='PhaseAverageReturn', buffer_size=np.inf),
            py_metrics.AverageEpisodeLengthMetric(
                name='PhaseAverageEpisodeLength', buffer_size=np.inf),
        ]
        self._iteration_metric = py_metrics.CounterMetric(name='Iteration')

        # Summaries written from python should run every time they are
        # generated.
        with tf.compat.v2.summary.record_if(True):
          self._steps_per_second_ph = tf.compat.v1.placeholder(
              tf.float32, shape=(), name='steps_per_sec_ph')
          self._steps_per_second_summary = tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=self._steps_per_second_ph,
              step=self._global_step)

          for metric in self._train_metrics:
            metric.tf_summaries(
                train_step=self._global_step, step_metrics=self._step_metrics)

          for metric in self._train_phase_metrics:
            metric.tf_summaries(
                train_step=self._global_step,
                step_metrics=(self._iteration_metric,))
          self._iteration_metric.tf_summaries(train_step=self._global_step)

          if self._do_eval:
            with self._eval_summary_writer.as_default():
              for metric in self._eval_metrics:
                metric.tf_summaries(
                    train_step=self._global_step,
                    step_metrics=(self._iteration_metric,))

        self._train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=agent,
            global_step=self._global_step,
            optimizer=optimizer,
            metrics=metric_utils.MetricsGroup(
                self._train_metrics + self._train_phase_metrics +
                [self._iteration_metric], 'train_metrics'))
        self._policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=agent.policy,
            global_step=self._global_step)
        self._rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=self._replay_buffer)

        self._init_agent_op = agent.initialize()

  def game_over(self):
    return self._env.envs[0].game_over

  def run(self):
    """Execute the train/eval loop."""
    with tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
      # Initialize the graph.
      self._initialize_graph(sess)

      # Initial collect
      self._initial_collect()

      while self._iteration_metric.result() < self._num_iterations:
        # Train phase
        env_steps = 0
        for metric in self._train_phase_metrics:
          metric.reset()
        while env_steps < self._train_steps_per_iteration:
          env_steps += self._run_episode(
              sess, self._train_metrics + self._train_phase_metrics, train=True)
        for metric in self._train_phase_metrics:
          log_metric(metric, prefix='Train/Metrics')
        py_metric.run_summaries(
            self._train_phase_metrics + [self._iteration_metric])

        global_step_val = sess.run(self._global_step)

        if self._do_eval:
          # Eval phase
          env_steps = 0
          for metric in self._eval_metrics:
            metric.reset()
          while env_steps < self._eval_steps_per_iteration:
            env_steps += self._run_episode(
                sess, self._eval_metrics, train=False)

          py_metric.run_summaries(self._eval_metrics + [self._iteration_metric])
          if self._eval_metrics_callback:
            results = dict((metric.name, metric.result())
                           for metric in self._eval_metrics)
            self._eval_metrics_callback(results, global_step_val)
          for metric in self._eval_metrics:
            log_metric(metric, prefix='Eval/Metrics')

        self._iteration_metric()

        self._train_checkpointer.save(global_step=global_step_val)
        self._policy_checkpointer.save(global_step=global_step_val)
        self._rb_checkpointer.save(global_step=global_step_val)

  def _initialize_graph(self, sess):
    """Initialize the graph for sess."""
    self._train_checkpointer.initialize_or_restore(sess)
    self._rb_checkpointer.initialize_or_restore(sess)
    common.initialize_uninitialized_variables(sess)

    sess.run(self._init_agent_op)

    self._train_step_call = sess.make_callable(self._train_op)

    self._collect_timer = timer.Timer()
    self._train_timer = timer.Timer()
    self._action_timer = timer.Timer()
    self._step_timer = timer.Timer()
    self._observer_timer = timer.Timer()

    global_step_val = sess.run(self._global_step)
    self._timed_at_step = global_step_val

    # Call save to initialize the save_counter (need to do this before
    # finalizing the graph).
    self._train_checkpointer.save(global_step=global_step_val)
    self._policy_checkpointer.save(global_step=global_step_val)
    self._rb_checkpointer.save(global_step=global_step_val)
    sess.run(self._train_summary_writer.init())

    if self._do_eval:
      sess.run(self._eval_summary_writer.init())

  def _initial_collect(self):
    """Collect initial experience before training begins."""
    logging.info('Collecting initial experience...')
    time_step_spec = ts.time_step_spec(self._env.observation_spec())
    random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec, self._env.action_spec())
    time_step = self._env.reset()
    while self._replay_buffer.size < self._initial_collect_steps:
      if self.game_over():
        time_step = self._env.reset()
      action_step = random_policy.action(time_step)
      next_time_step = self._env.step(action_step.action)
      self._replay_buffer.add_batch(trajectory.from_transition(
          time_step, action_step, next_time_step))
      time_step = next_time_step
    logging.info('Done.')

  def _run_episode(self, sess, metric_observers, train=False):
    """Run a single episode."""
    env_steps = 0
    time_step = self._env.reset()
    while True:
      with self._collect_timer:
        time_step = self._collect_step(
            time_step,
            metric_observers,
            train=train)
        env_steps += 1

      if self.game_over():
        break
      elif train and self._env_steps_metric.result() % self._update_period == 0:
        with self._train_timer:
          total_loss = self._train_step_call()
          global_step_val = sess.run(self._global_step)
        self._maybe_log(sess, global_step_val, total_loss)
        self._maybe_record_summaries(global_step_val)

    return env_steps

  def _observe(self, metric_observers, traj):
    with self._observer_timer:
      for observer in metric_observers:
        observer(traj)

  def _store_to_rb(self, traj):
    # Clip the reward to (-1, 1) to normalize rewards in training.
    traj = traj._replace(
        reward=np.asarray(np.clip(traj.reward, -1, 1)))
    self._replay_buffer.add_batch(traj)

  def _collect_step(self, time_step, metric_observers, train=False):
    """Run a single step (or 2 steps on life loss) in the environment."""
    if train:
      policy = self._collect_policy
    else:
      policy = self._eval_policy

    with self._action_timer:
      action_step = policy.action(time_step)
    with self._step_timer:
      next_time_step = self._env.step(action_step.action)
      traj = trajectory.from_transition(time_step, action_step, next_time_step)

    if next_time_step.is_last() and not self.game_over():
      traj = traj._replace(discount=np.array([1.0], dtype=np.float32))

    if train:
      self._store_to_rb(traj)

    # When AtariPreprocessing.terminal_on_life_loss is True, we receive LAST
    # time_steps when lives are lost but the game is not over.In this mode, the
    # replay buffer and agent's policy must see the life loss as a LAST step
    # and the subsequent step as a FIRST step. However, we do not want to
    # actually terminate the episode and metrics should be computed as if all
    # steps were MID steps, since life loss is not actually a terminal event
    # (it is mostly a trick to make it easier to propagate rewards backwards by
    # shortening episode durations from the agent's perspective).
    if next_time_step.is_last() and not self.game_over():
      # Update metrics as if this is a mid-episode step.
      next_time_step = ts.transition(
          next_time_step.observation, next_time_step.reward)
      self._observe(metric_observers, trajectory.from_transition(
          time_step, action_step, next_time_step))

      # Produce the next step as if this is the first step of an episode and
      # store to RB as such. The next_time_step will be a MID time step.
      reward = time_step.reward
      time_step = ts.restart(next_time_step.observation)
      with self._action_timer:
        action_step = policy.action(time_step)
      with self._step_timer:
        next_time_step = self._env.step(action_step.action)
      if train:
        self._store_to_rb(trajectory.from_transition(
            time_step, action_step, next_time_step))

      # Update metrics as if this is a mid-episode step.
      time_step = ts.transition(time_step.observation, reward)
      traj = trajectory.from_transition(time_step, action_step, next_time_step)

    self._observe(metric_observers, traj)

    return next_time_step

  def _maybe_record_summaries(self, global_step_val):
    """Record summaries if global_step_val is a multiple of summary_interval."""
    if global_step_val % self._summary_interval == 0:
      py_metric.run_summaries(self._train_metrics)

  def _maybe_log(self, sess, global_step_val, total_loss):
    """Log some stats if global_step_val is a multiple of log_interval."""
    if global_step_val % self._log_interval == 0:
      logging.info('step = %d, loss = %f', global_step_val, total_loss.loss)
      logging.info('%s', 'action_time = {}'.format(self._action_timer.value()))
      logging.info('%s', 'step_time = {}'.format(self._step_timer.value()))
      logging.info('%s', 'oberver_time = {}'.format(
          self._observer_timer.value()))
      steps_per_sec = ((global_step_val - self._timed_at_step) /
                       (self._collect_timer.value()
                        + self._train_timer.value()))
      sess.run(self._steps_per_second_summary,
               feed_dict={self._steps_per_second_ph: steps_per_sec})
      logging.info('%.3f steps/sec', steps_per_sec)
      logging.info('%s', 'collect_time = {}, train_time = {}'.format(
          self._collect_timer.value(), self._train_timer.value()))
      for metric in self._train_metrics:
        log_metric(metric, prefix='Train/Metrics')
      self._timed_at_step = global_step_val
      self._collect_timer.reset()
      self._train_timer.reset()
      self._action_timer.reset()
      self._step_timer.reset()
      self._observer_timer.reset()


def get_run_args():
  """Builds a dict of run arguments from flags."""
  run_args = {}
  if FLAGS.num_iterations:
    run_args['num_iterations'] = FLAGS.num_iterations
  if FLAGS.initial_collect_steps:
    run_args['initial_collect_steps'] = FLAGS.initial_collect_steps
  if FLAGS.replay_buffer_capacity:
    run_args['replay_buffer_capacity'] = FLAGS.replay_buffer_capacity
  if FLAGS.train_steps_per_iteration:
    run_args['train_steps_per_iteration'] = FLAGS.train_steps_per_iteration
  if FLAGS.n_step_update:
    run_args['n_step_update'] = FLAGS.n_step_update
  if FLAGS.eval_steps_per_iteration:
    run_args['eval_steps_per_iteration'] = FLAGS.eval_steps_per_iteration
  return run_args


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_resource_variables()
  TrainEval(FLAGS.root_dir, suite_atari.game(name=FLAGS.game_name),
            **get_run_args()).run()


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
