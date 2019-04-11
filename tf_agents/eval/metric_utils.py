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

"""Utils for Metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging

import tensorflow as tf
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import py_driver
from tf_agents.metrics import py_metric
from tf_agents.utils import common


class MetricsGroup(tf.Module):
  """Group a list of Metrics into a container."""

  def __init__(self, metrics, name=None):
    super(MetricsGroup, self).__init__(name=name)
    self.metrics = metrics

  def results(self):
    results = [(metric.name, metric.result()) for metric in self.metrics]
    return collections.OrderedDict(results)


def log_metrics(metrics, prefix=''):
  log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
  logging.info('%s \n\t\t %s', prefix, '\n\t\t '.join(log))


def compute(metrics,
            environment,
            policy,
            num_episodes=1):
  """Compute metrics using `policy` on the `environment`.

  Args:
    metrics: List of metrics to compute.
    environment: py_environment instance.
    policy: py_policy instance used to step the environment. A tf_policy can be
      used in_eager_mode.
    num_episodes: Number of episodes to compute the metrics over.

  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  for metric in metrics:
    metric.reset()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  driver = py_driver.PyDriver(
      environment,
      policy,
      observers=metrics,
      max_steps=None,
      max_episodes=num_episodes)
  driver.run(time_step, policy_state)

  results = [(metric.name, metric.result()) for metric in metrics]
  return collections.OrderedDict(results)


def compute_summaries(metrics,
                      environment,
                      policy,
                      num_episodes=1,
                      global_step=None,
                      tf_summaries=True,
                      log=False,
                      callback=None):
  """Compute metrics using `policy` on the `environment` and logs summaries.

  Args:
    metrics: List of metrics to compute.
    environment: py_environment instance.
    policy: py_policy instance used to step the environment. A tf_policy can be
      used in_eager_mode.
    num_episodes: Number of episodes to compute the metrics over.
    global_step: An optional global step for summaries.
    tf_summaries: If True, write TF summaries for each computed metric.
    log: If True, log computed metrics.
    callback: If provided, this function is called with (computed_metrics,
      global_step).

  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  results = compute(metrics, environment, policy, num_episodes)
  if tf_summaries:
    py_metric.run_summaries(metrics)
  if log:
    log_metrics(metrics, prefix='Step = {}'.format(global_step))
  if callback is not None:
    callback(results, global_step)
  return results


# TODO(b/130250285): Match compute and compute_summaries signatures.
def eager_compute(metrics,
                  environment,
                  policy,
                  num_episodes=1,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix=''):
  """Compute metrics using `policy` on the `environment`.

  *NOTE*: Because placeholders are not compatible with Eager mode we can not use
  python policies. Because we use tf_policies we need the environment time_steps
  to be tensors making it easier to use a tf_env for evaluations. Otherwise this
  method mirrors `compute` directly.

  Args:
    metrics: List of metrics to compute.
    environment: tf_environment instance.
    policy: tf_policy instance used to step the environment.
    num_episodes: Number of episodes to compute the metrics over.
    train_step: An optional step to write summaries against.
    summary_writer: An optional writer for generating metric summaries.
    summary_prefix: An optional prefix scope for metric summaries.
  Returns:
    A dictionary of results {metric_name: metric_value}
  """
  for metric in metrics:
    metric.reset()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  driver = dynamic_episode_driver.DynamicEpisodeDriver(
      environment,
      policy,
      observers=metrics,
      num_episodes=num_episodes)
  common.function(driver.run)(time_step, policy_state)

  results = [(metric.name, metric.result()) for metric in metrics]
  # TODO(b/120301678) remove the summaries and merge with compute
  if train_step and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = common.join_scope(summary_prefix, m.name)
        tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)
  # TODO(b/130249101): Add an option to log metrics.
  return collections.OrderedDict(results)
