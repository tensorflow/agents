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

"""Tests for tf_agents.metrics.metric_equality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import batched_py_environment
from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.policies import random_tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class MetricEqualityTest(test_utils.TestCase):

  def _build_metrics(self, buffer_size=10, batch_size=None):
    python_metrics = [
        tf_py_metric.TFPyMetric(
            py_metrics.AverageReturnMetric(
                buffer_size=buffer_size, batch_size=batch_size)),
        tf_py_metric.TFPyMetric(
            py_metrics.AverageEpisodeLengthMetric(
                buffer_size=buffer_size, batch_size=batch_size)),
    ]
    if batch_size is None:
      batch_size = 1
    tensorflow_metrics = [
        tf_metrics.AverageReturnMetric(
            buffer_size=buffer_size, batch_size=batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=buffer_size, batch_size=batch_size),
    ]

    return python_metrics, tensorflow_metrics

  def setUp(self):
    super(MetricEqualityTest, self).setUp()
    observation_spec = array_spec.BoundedArraySpec((1,),
                                                   dtype=np.float32,
                                                   minimum=0,
                                                   maximum=10)
    self._action_spec = array_spec.BoundedArraySpec((1,),
                                                    dtype=np.float32,
                                                    minimum=0,
                                                    maximum=10)
    reward_spec = array_spec.BoundedArraySpec((),
                                              dtype=np.float32,
                                              minimum=0,
                                              maximum=10)
    time_step_spec = ts.time_step_spec(observation_spec)
    self._time_step_spec = time_step_spec._replace(reward=reward_spec)

    self._tensor_action_spec = tensor_spec.from_spec(self._action_spec)
    self._tensor_time_step_spec = tensor_spec.from_spec(self._time_step_spec)

    self._env = random_py_environment.RandomPyEnvironment(
        observation_spec, self._action_spec)
    self._tf_env = tf_py_environment.TFPyEnvironment(self._env)
    self._policy = random_tf_policy.RandomTFPolicy(self._tensor_time_step_spec,
                                                   self._tensor_action_spec)

  def test_metric_results_equal(self):
    python_metrics, tensorflow_metrics = self._build_metrics()
    observers = python_metrics + tensorflow_metrics
    driver = dynamic_step_driver.DynamicStepDriver(
        self._tf_env, self._policy, observers=observers, num_steps=1000)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(driver.run())

    for python_metric, tensorflow_metric in zip(python_metrics,
                                                tensorflow_metrics):
      python_result = self.evaluate(python_metric.result())
      tensorflow_result = self.evaluate(tensorflow_metric.result())
      self.assertEqual(python_result, tensorflow_result)

  def test_metric_results_equal_with_batched_env(self):
    env_ctor = lambda: random_py_environment.RandomPyEnvironment(  # pylint: disable=g-long-lambda
        self._time_step_spec.observation, self._action_spec)
    batch_size = 5
    env = batched_py_environment.BatchedPyEnvironment(
        [env_ctor() for _ in range(batch_size)])
    tf_env = tf_py_environment.TFPyEnvironment(env)

    python_metrics, tensorflow_metrics = self._build_metrics(
        batch_size=batch_size)
    observers = python_metrics + tensorflow_metrics
    driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, self._policy, observers=observers, num_steps=1000)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(driver.run())

    for python_metric, tensorflow_metric in zip(python_metrics,
                                                tensorflow_metrics):
      python_result = self.evaluate(python_metric.result())
      tensorflow_result = self.evaluate(tensorflow_metric.result())
      self.assertEqual(python_result, tensorflow_result)


if __name__ == '__main__':
  tf.compat.v1.enable_resource_variables()
  test_utils.main()
