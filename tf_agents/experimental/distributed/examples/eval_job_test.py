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
from tf_agents.train.utils import train_utils
from tf_agents.utils import test_utils


class _NTimesReturnTrue(object):
  """Statefull callable which return `True` N-times, then `False`."""

  def __init__(self, n=1):
    self._n = n

  def __call__(self):
    self._n -= 1
    return 0 <= self._n


class EvalJobTest(test_utils.TestCase):

  def test_eval_job(self):
    # Create test context.
    summary_dir = self.create_tempdir().full_path
    environment = test_envs.CountingEnv(steps_per_episode=4)
    action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
    time_step_tensor_spec = tensor_spec.from_spec(environment.time_step_spec())
    policy = py_tf_eager_policy.PyTFEagerPolicy(
        random_tf_policy.RandomTFPolicy(time_step_tensor_spec,
                                        action_tensor_spec))
    mock_variable_container = mock.create_autospec(
        reverb_variable_container.ReverbVariableContainer)

    with mock.patch.object(
        tf.summary, 'scalar',
        autospec=True) as mock_scalar_summary, mock.patch.object(
            train_utils, 'wait_for_predicate', autospec=True):
      # Run the function tested.
      eval_job.evaluate(
          summary_dir=summary_dir,
          policy=policy,
          environment_name=None,
          suite_load_fn=lambda _: environment,
          variable_container=mock_variable_container,
          is_running=_NTimesReturnTrue(n=2))

      # Check if the expected calls happened.
      # As an input, an eval job is expected to fetch data from the variable
      # container.
      mock_variable_container.assert_has_calls([mock.call.update(mock.ANY)])

      # As an output, an eval job is expected to write at least the average
      # return corresponding to the first step.
      mock_scalar_summary.assert_any_call(
          name='eval_actor/AverageReturn', data=mock.ANY, step=mock.ANY)


if __name__ == '__main__':
  multiprocessing.handle_test_main(test_utils.main)
