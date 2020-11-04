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
"""Actor to use for data collection or evaluation.

**Note** the actor currently only supports py_envs, policies/drivers.
"""
import os

from absl import logging
import gin
import tensorflow.compat.v2 as tf

from tf_agents.drivers import py_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.metrics import py_metrics
from tf_agents.utils import common


@gin.configurable
class Actor(object):
  """Actor.

  The actor manages interactions between a policy and an environment. Users
  should configure the metrics and summaries for a specific task like evaluation
  or data collection.

  The main point of access for users is  the `run` method. This will iterate
  over either n `steps_per_run` or `episodes_per_run`. At least one of
  `steps_per_run` or `episodes_per_run` must be provided.
  """

  def __init__(self,
               env,
               policy,
               train_step,
               steps_per_run=None,
               episodes_per_run=None,
               observers=None,
               metrics=None,
               reference_metrics=None,
               summary_dir=None,
               summary_interval=1000,
               name=""):
    """Initializes an Actor.

    Args:
      env: An instance of either a tf or py environment. Note the policy, and
        observers should match the tf/pyness of the env.
      policy: An instance of a policy used to interact with the environment.
      train_step: A scalar tf.int64 `tf.Variable` which will keep track of the
        number of train steps. This is used for artifacts created like
        summaries.
      steps_per_run: Number of steps to evaluated per run call. See below.
      episodes_per_run: Number of episodes evaluated per run call.
      observers: A list of observers that are notified after every step in the
        environment. Each observer is a callable(trajectory.Trajectory).
      metrics: A list of metric observers.
      reference_metrics: Optional list of metrics for which other metrics are
        plotted against. As an example passing in a metric that tracks number of
        environment episodes will result in having summaries of all other
        metrics over this value. Note summaries against the train_step are done
        by default. If you want reference_metrics to be updated make sure they
        are also added to the metrics list.
      summary_dir: Path used for summaries. If no path is provided no summaries
        are written.
      summary_interval: How often summaries are written.
      name: Name for the actor used as a prefix to generated summaries.
    """
    self._env = env
    self._policy = policy
    self._train_step = train_step
    self._observers = observers or []
    self._metrics = metrics or []
    self._observers.extend(self._metrics)
    self._reference_metrics = reference_metrics or []
    # Make sure metrics are not repeated.
    self._observers = list(set(self._observers))

    self._write_summaries = bool(summary_dir)  # summary_dir is not None

    if self._write_summaries:
      self._summary_writer = tf.summary.create_file_writer(
          summary_dir, flush_millis=10000)
    else:
      self._summary_writer = NullSummaryWriter()

    self._summary_interval = summary_interval
    # In order to write summaries at `train_step=0` as well.
    self._last_summary = -summary_interval

    self._name = name

    if isinstance(env, py_environment.PyEnvironment):
      self._driver = py_driver.PyDriver(
          env,
          policy,
          self._observers,
          max_steps=steps_per_run,
          max_episodes=episodes_per_run)
    elif isinstance(env, tf_environment.TFEnvironment):
      raise ValueError("Actor doesn't support TFEnvironments yet.")
    else:
      raise ValueError("Unknown environment type.")

    self.reset()

  @property
  def metrics(self):
    return self._metrics

  @property
  def summary_writer(self):
    return self._summary_writer

  @property
  def train_step(self):
    return self._train_step

  @property
  def policy(self):
    return self._policy

  def run(self):
    self._time_step, self._policy_state = self._driver.run(
        self._time_step, self._policy_state)

    if (self._write_summaries and self._summary_interval > 0 and
        self._train_step - self._last_summary >= self._summary_interval):
      self.write_metric_summaries()
      self._last_summary = self._train_step.numpy()

  def run_and_log(self):
    self.run()
    self.log_metrics()

  def write_metric_summaries(self):
    """Generates scalar summaries for the actor metrics."""
    if self._metrics is None:
      return
    with self._summary_writer.as_default(), \
         common.soft_device_placement(), \
         tf.summary.record_if(lambda: True):
      # Generate summaries against the train_step
      for m in self._metrics:
        tag = m.name
        tf.summary.scalar(
            name=os.path.join(self._name, tag),
            data=m.result(),
            step=self._train_step)
        # Generate summaries against the reference_metrics
        for reference_metric in self._reference_metrics:
          tag = "{}/{}".format(m.name, reference_metric.name)
          tf.summary.scalar(
              name=os.path.join(self._name, tag),
              data=m.result(),
              step=reference_metric.result())

  def log_metrics(self):
    """Logs metric results to stdout."""
    if self._metrics is None:
      return
    log = ["{0} = {1}".format(m.name, m.result()) for m in self._metrics]
    logging.info("%s \n\t\t %s", self._name, "\n\t\t ".join(log))

  def reset(self):
    """Reset the environment to the start and the policy state."""
    self._time_step = self._env.reset()
    self._policy_state = self._policy.get_initial_state(
        self._env.batch_size or 1)


def collect_metrics(buffer_size):
  """Utilitiy to create metrics often used during data collection."""
  metrics = [
      py_metrics.NumberOfEpisodes(),
      py_metrics.EnvironmentSteps(),
      py_metrics.AverageReturnMetric(buffer_size=buffer_size),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=buffer_size),
  ]
  return metrics


def eval_metrics(buffer_size):
  """Utilitiy to create metrics often used during policy evaluation."""
  return [
      py_metrics.AverageReturnMetric(buffer_size=buffer_size),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=buffer_size),
  ]


class NullSummaryWriter(object):
  """Class to fake a context manager object when no SummaryWriter is needed."""

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass
