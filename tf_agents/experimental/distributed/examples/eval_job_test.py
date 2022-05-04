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

"""Tests the distributed eval jobs."""

from absl.testing.absltest import mock
import tensorflow.compat.v2 as tf

from tf_agents.environments import test_envs
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.experimental.distributed.examples import eval_job
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import test_utils


class _NTimesReturnTrue(object):
  """Stateful callable which return `True` N-times, then `False`."""

  def __init__(self, n=1):
    self._n = n

  def __call__(self):
    self._n -= 1
    return 0 <= self._n


class EvalJobTest(test_utils.TestCase):

  def test_eval_job(self):
    """Tests the eval job doing an eval every 5 steps for 10 train steps."""
    summary_dir = self.create_tempdir().full_path
    environment = test_envs.CountingEnv(steps_per_episode=4)
    action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
    time_step_tensor_spec = tensor_spec.from_spec(environment.time_step_spec())
    policy = py_tf_eager_policy.PyTFEagerPolicy(
        random_tf_policy.RandomTFPolicy(time_step_tensor_spec,
                                        action_tensor_spec))

    class VCUpdateIncrementTrainStep(object):
      """Side effect that updates train_step."""

      def __init__(self):
        self.fake_train_step = -1

      def __call__(self, variables):
        self.fake_train_step += 1
        variables[reverb_variable_container.TRAIN_STEP_KEY].assign(
            self.fake_train_step)

    mock_variable_container = mock.create_autospec(
        reverb_variable_container.ReverbVariableContainer)
    fake_update = VCUpdateIncrementTrainStep()
    mock_variable_container.update.side_effect = fake_update

    with mock.patch.object(
        tf.summary, 'scalar', autospec=True) as mock_scalar_summary:
      # Run the function tested.
      # 11 loops to do 10 steps becaue the eval occurs on the loop after the
      # train_step is found.
      eval_job.evaluate(
          summary_dir=summary_dir,
          policy=policy,
          environment_name=None,
          suite_load_fn=lambda _: environment,
          variable_container=mock_variable_container,
          eval_interval=5,
          is_running=_NTimesReturnTrue(n=11))

      summary_count = self.count_summary_scalar_tags_in_call_list(
          mock_scalar_summary, 'Metrics/eval_actor/AverageReturn')
      self.assertEqual(summary_count, 3)

  def test_eval_job_constant_eval(self):
    """Tests eval every step for 2 steps.

    This test's `variable_container` passes the same train step twice to test
    that `is_train_step_the_same_or_behind` is working as expected. If were not
    working, the number of train steps processed will be incorrect (2x higher).
    """
    summary_dir = self.create_tempdir().full_path
    environment = test_envs.CountingEnv(steps_per_episode=4)
    action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
    time_step_tensor_spec = tensor_spec.from_spec(environment.time_step_spec())
    policy = py_tf_eager_policy.PyTFEagerPolicy(
        random_tf_policy.RandomTFPolicy(time_step_tensor_spec,
                                        action_tensor_spec))
    mock_variable_container = mock.create_autospec(
        reverb_variable_container.ReverbVariableContainer)

    class VCUpdateIncrementEveryOtherTrainStep(object):
      """Side effect that updates train_step on every other call."""

      def __init__(self):
        self.fake_train_step = -1
        self.call_count = 0

      def __call__(self, variables):
        if self.call_count % 2:
          self.fake_train_step += 1
          variables[reverb_variable_container.TRAIN_STEP_KEY].assign(
              self.fake_train_step)
        self.call_count += 1

    fake_update = VCUpdateIncrementEveryOtherTrainStep()
    mock_variable_container.update.side_effect = fake_update

    with mock.patch.object(
        tf.summary, 'scalar', autospec=True) as mock_scalar_summary:
      eval_job.evaluate(
          summary_dir=summary_dir,
          policy=policy,
          environment_name=None,
          suite_load_fn=lambda _: environment,
          variable_container=mock_variable_container,
          eval_interval=1,
          is_running=_NTimesReturnTrue(n=2))

      summary_count = self.count_summary_scalar_tags_in_call_list(
          mock_scalar_summary, 'Metrics/eval_actor/AverageReturn')
      self.assertEqual(summary_count, 2)

  def count_summary_scalar_tags_in_call_list(self, mock_summary_scalar, tag):
    """Returns the number of time the tag is found in `mock_summary_scalar`.

    This is used because `assert_has_calls` uses a list for verification that
    is cumbersome and produces confusing error messages on unit test failure.
    Example: Index out of bounds if more values exist than expected. This is not
    intutive compared with counting the items in the list.
    To debug: `print('{}'.format(mock_scalar_summary.mock_calls))`.

    Args:
      mock_summary_scalar: Name of the summary tag.
      tag: Name of the summary tag to count.

    Returns:
      Number of times the tag was added via the `mock_summary_scalar`.
    """
    # The `call` object is a tuple. [2] is a dictionary of the keyword args.
    return sum(1 for c in mock_summary_scalar.mock_calls if c[2]['name'] == tag)


if __name__ == '__main__':
  multiprocessing.handle_test_main(test_utils.main)
