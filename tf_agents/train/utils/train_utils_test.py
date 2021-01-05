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
"""Tests for distributed training utils of the Actor/Learner API."""

from absl.testing import parameterized
from absl.testing.absltest import mock

import numpy as np

import reverb

import tensorflow as tf

from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import test_utils as dist_test_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import test_utils

_CPUS = ('/cpu:0', '/cpu:1', '/cpu:2', '/cpu:3')
_TESTS = (('_default', tf.distribute.get_strategy),
          ('_one_device', lambda: tf.distribute.OneDeviceStrategy('/cpu:0')),
          ('_mirrored', lambda: tf.distribute.MirroredStrategy(devices=_CPUS)))


class TrainUtilsTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(TrainUtilsTest, self).setUp()
    dist_test_utils.configure_logical_cpus()

  @parameterized.named_parameters(*_TESTS)
  def test_after_train_step_fn_with_fresh_data_only(self, create_strategy_fn):
    strategy = create_strategy_fn()
    with strategy.scope():
      # Prepare the test context context.
      train_step = train_utils.create_train_step()
      train_step.assign(225)
      train_steps_per_policy_update = 100

      # Create the after train function to test, and the test input.
      after_train_step_fn = (
          train_utils.create_staleness_metrics_after_train_step_fn(
              train_step,
              train_steps_per_policy_update=train_steps_per_policy_update))
      observation_train_steps = np.array([[200], [200], [200]], dtype=np.int64)

      # Define the expectations (expected scalar summary calls).
      expected_scalar_summary_calls = [
          mock.call(
              name='staleness/max_train_step_delta_in_batch', data=0, step=225),
          mock.call(
              name='staleness/max_policy_update_delta_in_batch',
              data=0,
              step=225),
          mock.call(
              name='staleness/num_stale_obserations_in_batch', data=0, step=225)
      ]

      # Call the after train function and check the expectations.
      with mock.patch.object(
          tf.summary, 'scalar', autospec=True) as mock_scalar_summary:
        # Call the `after_train_function` on the test input. Assumed the
        # observation train steps are stored in the field `priority` of the
        # the sample info of Reverb.
        strategy.run(
            after_train_step_fn,
            args=((None,
                   reverb.replay_sample.SampleInfo(
                       key=None,
                       probability=None,
                       table_size=None,
                       priority=observation_train_steps)), None))

        # Check if the expected calls happened on the scalar summary.
        mock_scalar_summary.assert_has_calls(
            expected_scalar_summary_calls, any_order=False)

  @parameterized.named_parameters(*_TESTS)
  def test_after_train_step_fn_with_stale_data(self, create_strategy_fn):
    strategy = create_strategy_fn()
    with strategy.scope():
      # Prepare the test context context.
      train_step = train_utils.create_train_step()
      train_step.assign(225)
      train_steps_per_policy_update = 100

      # Create the after train function to test, and the test input.
      after_train_step_fn = (
          train_utils.create_staleness_metrics_after_train_step_fn(
              train_step,
              train_steps_per_policy_update=train_steps_per_policy_update))
      observation_train_steps = np.array([[100], [200]], dtype=np.int64)

      # Define the expectations (expected scalar summary calls).
      expected_scalar_summary_calls = [
          mock.call(
              name='staleness/max_train_step_delta_in_batch',
              data=100,
              step=225),
          mock.call(
              name='staleness/max_policy_update_delta_in_batch',
              data=1,
              step=225),
          mock.call(
              name='staleness/num_stale_obserations_in_batch', data=1, step=225)
      ]

      # Call the after train function and check the expectations.
      with mock.patch.object(
          tf.summary, 'scalar', autospec=True) as mock_scalar_summary:
        # Call the `after_train_function` on the test input. Assumed the
        # observation train steps are stored in the field `priority` of the
        # the sample info of Reverb.
        strategy.run(
            after_train_step_fn,
            args=((None,
                   reverb.replay_sample.SampleInfo(
                       key=None,
                       probability=None,
                       table_size=None,
                       priority=observation_train_steps)), None))

        # Check if the expected calls happened on the scalar summary.
        mock_scalar_summary.assert_has_calls(
            expected_scalar_summary_calls, any_order=False)

  def test_wait_for_predicate_instant_false(self):
    """Tests predicate returning False on first call."""
    predicate_mock = mock.MagicMock(side_effect=[False])
    # 10 retry limit to avoid a near infinite loop on an error.
    train_utils.wait_for_predicate(predicate_mock, num_retries=10)
    self.assertEqual(predicate_mock.call_count, 1)

  def test_wait_for_predicate_second_false(self):
    """Tests predicate returning False on second call."""
    predicate_mock = mock.MagicMock(side_effect=[True, False])
    # 10 retry limit to avoid a near infinite loop on an error.
    train_utils.wait_for_predicate(predicate_mock, num_retries=10)
    self.assertEqual(predicate_mock.call_count, 2)

  def test_wait_for_predicate_timeout(self):
    """Tests predicate returning True forever and then timing out."""
    predicate_mock = mock.MagicMock(side_effect=[True, True, True])
    with self.assertRaises(TimeoutError):
      train_utils.wait_for_predicate(predicate_mock, num_retries=3)


if __name__ == '__main__':
  multiprocessing.handle_test_main(test_utils.main)
