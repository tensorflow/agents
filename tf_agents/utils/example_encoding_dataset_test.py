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

"""Tests for example_encoding_dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import py_driver
from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.drivers import tf_driver
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import example_encoding_dataset
from tf_agents.utils import test_utils

SimpleSpec = collections.namedtuple("SimpleSpec", ("step_type", "value"))


class TFRecordsUtilsTest(test_utils.TestCase):
  """Tests for TFRecordObserver class and dataset/spec file functions."""

  def setUp(self):
    super(TFRecordsUtilsTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    # Create a simple mock tensor data spec for testing
    # Spec has a batch dimmension of 1 to simulate environment samples
    self.simple_data_spec = SimpleSpec(
        step_type=tf.TensorSpec(shape=(1), dtype=tf.int32, name="step_type"),
        value=tf.TensorSpec(shape=(1, 2), dtype=tf.float64, name="value"))
    self.dataset_path = os.path.join(self.get_temp_dir(),
                                     "test_tfrecord_dataset.tfrecord")

  def test_tfrecord_observer(self):
    tfrecord_observer = example_encoding_dataset.TFRecordObserver(
        self.dataset_path, self.simple_data_spec)
    # Draw a random sample from the simple spec
    sample = tensor_spec.sample_spec_nest(
        self.simple_data_spec, np.random.RandomState(0), outer_dims=(1,))
    # Write to file using __call__() function
    for _ in range(3):
      tfrecord_observer(sample)
    # Manually flush
    tfrecord_observer.flush()
    # Delete should call close() function
    del tfrecord_observer

  def test_load_tfrecord_dataset(self):
    # Make sure an example tfrecord file exists before attempting to load
    self.test_tfrecord_observer()
    example_encoding_dataset.load_tfrecord_dataset([self.dataset_path],
                                                   buffer_size=2)
    example_encoding_dataset.load_tfrecord_dataset([self.dataset_path],
                                                   buffer_size=2,
                                                   as_experience=True)
    with self.assertRaises(IOError):
      example_encoding_dataset.load_tfrecord_dataset(["fake_file.tfrecord"])

  def test_spec_to_from_file(self):
    example_encoding_dataset.encode_spec_to_file(self.dataset_path,
                                                 self.simple_data_spec)
    loaded_spec = example_encoding_dataset.parse_encoded_spec_from_file(
        self.dataset_path)
    self.assertTupleEqual(loaded_spec, self.simple_data_spec)
    with self.assertRaises(IOError):
      example_encoding_dataset.parse_encoded_spec_from_file(
          "fake_file.tfrecord")

  def test_conflicting_specs(self):
    # If two different specs are encountered an error should be thrown
    self.other_data_spec = SimpleSpec(
        step_type=tf.TensorSpec(shape=(1), dtype=tf.int32, name="step_type"),
        value=tf.TensorSpec(shape=(1, 5), dtype=tf.float64, name="value"))
    self.other_dataset_path = os.path.join(
        self.get_temp_dir(), "other_test_tfrecord_dataset.tfrecord")
    example_encoding_dataset.encode_spec_to_file(self.other_dataset_path,
                                                 self.other_data_spec)
    with self.assertRaises(IOError):
      example_encoding_dataset.load_tfrecord_dataset(
          [self.dataset_path, self.other_dataset_path])

  def test_with_tf_driver(self):
    env = driver_test_utils.PyEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    policy = driver_test_utils.TFPolicyMock(tf_env.time_step_spec(),
                                            tf_env.action_spec())

    trajectory_spec = trajectory.from_transition(tf_env.time_step_spec(),
                                                 policy.policy_step_spec,
                                                 tf_env.time_step_spec())

    tfrecord_observer = example_encoding_dataset.TFRecordObserver(
        self.dataset_path, trajectory_spec)
    driver = tf_driver.TFDriver(
        tf_env, policy, [tfrecord_observer], max_steps=10)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    time_step = self.evaluate(tf_env.reset())
    initial_policy_state = policy.get_initial_state(batch_size=1)
    self.evaluate(common.function(driver.run)(time_step, initial_policy_state))
    tfrecord_observer.flush()

    dataset = example_encoding_dataset.load_tfrecord_dataset(
        [self.dataset_path], buffer_size=2, as_trajectories=True)
    iterator = eager_utils.dataset_iterator(dataset)
    sample = self.evaluate(eager_utils.get_next(iterator))
    self.assertIsInstance(sample, trajectory.Trajectory)

  def test_with_py_driver(self):
    env = driver_test_utils.PyEnvironmentMock()
    policy = driver_test_utils.PyPolicyMock(env.time_step_spec(),
                                            env.action_spec())
    trajectory_spec = trajectory.from_transition(env.time_step_spec(),
                                                 policy.policy_step_spec,
                                                 env.time_step_spec())
    trajectory_spec = tensor_spec.from_spec(trajectory_spec)

    tfrecord_observer = example_encoding_dataset.TFRecordObserver(
        self.dataset_path, trajectory_spec, py_mode=True)

    driver = py_driver.PyDriver(env, policy, [tfrecord_observer], max_steps=10)
    time_step = env.reset()
    driver.run(time_step)
    tfrecord_observer.flush()

    dataset = example_encoding_dataset.load_tfrecord_dataset(
        [self.dataset_path], buffer_size=2, as_trajectories=True)

    iterator = eager_utils.dataset_iterator(dataset)
    sample = self.evaluate(eager_utils.get_next(iterator))
    self.assertIsInstance(sample, trajectory.Trajectory)

  def test_with_dynamic_step_driver(self):
    env = driver_test_utils.PyEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    policy = driver_test_utils.TFPolicyMock(tf_env.time_step_spec(),
                                            tf_env.action_spec())

    trajectory_spec = trajectory.from_transition(tf_env.time_step_spec(),
                                                 policy.policy_step_spec,
                                                 tf_env.time_step_spec())

    tfrecord_observer = example_encoding_dataset.TFRecordObserver(
        self.dataset_path, trajectory_spec)
    driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        policy,
        observers=[common.function(tfrecord_observer)],
        num_steps=10)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    time_step = self.evaluate(tf_env.reset())
    initial_policy_state = policy.get_initial_state(batch_size=1)
    self.evaluate(common.function(driver.run)(time_step, initial_policy_state))
    tfrecord_observer.flush()

    dataset = example_encoding_dataset.load_tfrecord_dataset(
        [self.dataset_path], buffer_size=2, as_trajectories=True)
    iterator = eager_utils.dataset_iterator(dataset)
    sample = self.evaluate(eager_utils.get_next(iterator))
    self.assertIsInstance(sample, trajectory.Trajectory)


if __name__ == "__main__":
  tf.test.main()
