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

"""Tests for tf_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.utils import common as common


class TfPolicyHoldsVariables(tf_policy.Base):
  """Test tf_policy which contains only trainable variables."""

  def __init__(self, init_var_value, var_scope, name=None):
    """Initializes policy containing variables with specified value.

    Args:
      init_var_value: A scalar specifies the initial value of all variables.
      var_scope: A String defines variable scope.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)
    with tf.compat.v1.variable_scope(var_scope):
      self._variables_list = [
          common.create_variable("var_1", init_var_value, [3, 3],
                                 dtype=tf.float32),
          common.create_variable("var_2", init_var_value, [5, 5],
                                 dtype=tf.float32)
      ]

  def _variables(self):
    return self._variables_list

  def _action(self, time_step, policy_state, seed):
    pass

  def _distribution(self, time_step, policy_state):
    pass


class TfPolicyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("SoftUpdate", 0.5, False),
      ("SyncVariables", 1.0, True),
  )
  def testUpdate(self, tau, sort_variables_by_name):
    source_policy = TfPolicyHoldsVariables(init_var_value=1.,
                                           var_scope="source")
    target_policy = TfPolicyHoldsVariables(init_var_value=0.,
                                           var_scope="target")

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for var in self.evaluate(target_policy.variables()):
      self.assertAllEqual(var, np.zeros(var.shape))

    update_op = target_policy.update(
        source_policy, tau=tau, sort_variables_by_name=sort_variables_by_name)
    self.evaluate(update_op)
    for var in self.evaluate(target_policy.variables()):
      self.assertAllEqual(var, np.ones(var.shape)*tau)


if __name__ == "__main__":
  tf.test.main()
