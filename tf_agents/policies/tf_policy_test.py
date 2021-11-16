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

"""Tests for tf_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import cast

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils


class TfPolicyHoldsVariables(tf_policy.TFPolicy):
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
          common.create_variable(
              "var_1", init_var_value, [3, 3], dtype=tf.float32),
          common.create_variable(
              "var_2", init_var_value, [5, 5], dtype=tf.float32)
      ]

  def _variables(self):
    return self._variables_list

  def _action(self, time_step, policy_state, seed):
    return policy_step.PolicyStep(())

  def _distribution(self, time_step, policy_state):
    return policy_step.PolicyStep(())


class TFPolicyMismatchedDtypes(tf_policy.TFPolicy):
  """Dummy tf_policy with mismatched dtypes."""

  def __init__(self):
    observation_spec = tensor_spec.TensorSpec([2, 2], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    super(TFPolicyMismatchedDtypes, self).__init__(time_step_spec, action_spec)

  def _action(self, time_step, policy_state, seed):
    # This action's dtype intentionally doesn't match action_spec's dtype.
    return policy_step.PolicyStep(action=tf.constant([0], dtype=tf.int64))

  def _distribution(self, time_step, policy_state):
    return policy_step.PolicyStep(())


class TFPolicyMismatchedDtypesListAction(tf_policy.TFPolicy):
  """Dummy tf_policy with mismatched dtypes and a list action_spec."""

  def __init__(self):
    observation_spec = tensor_spec.TensorSpec([2, 2], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = [
        tensor_spec.BoundedTensorSpec([1], tf.int64, 0, 1),
        tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    ]
    super(TFPolicyMismatchedDtypesListAction,
          self).__init__(time_step_spec, action_spec)

  def _action(self, time_step, policy_state, seed):
    # This time, the action is a list where only the second dtype doesn't match.
    return policy_step.PolicyStep(action=[
        tf.constant([0], dtype=tf.int64),
        tf.constant([0], dtype=tf.int64)
    ])

  def _distribution(self, time_step, policy_state):
    return policy_step.PolicyStep(())


class TfPassThroughPolicy(tf_policy.TFPolicy):

  def _action(self, time_step, policy_state, seed):
    distributions = self._distribution(time_step, policy_state)
    actions = tf.nest.map_structure(lambda d: d.sample(), distributions.action)
    return policy_step.PolicyStep(actions, policy_state, ())

  def _distribution(self, time_step, policy_state):
    action_distribution = tf.nest.map_structure(
        lambda loc: tfp.distributions.Deterministic(loc=loc),
        time_step.observation)
    return policy_step.PolicyStep(action_distribution, policy_state, ())


class TfEmitLogProbsPolicy(tf_policy.TFPolicy):
  """Dummy policy with constant probability distribution."""

  def __init__(self, info_spec=()):
    observation_spec = tensor_spec.TensorSpec([2, 2], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 5)
    super(TfEmitLogProbsPolicy, self).__init__(
        time_step_spec,
        action_spec,
        info_spec=info_spec,
        emit_log_probability=True)

  def _distribution(self, time_step, policy_state):
    action_spec = cast(tensor_spec.BoundedTensorSpec, self.action_spec)
    probs = tf.constant(
        0.2, shape=[action_spec.maximum - action_spec.minimum])
    action_distribution = tf.nest.map_structure(
        lambda obs: tfp.distributions.Categorical(probs=probs),
        time_step.observation)
    step = policy_step.PolicyStep(action_distribution)
    return step


class TfDictInfoAndLogProbs(TfEmitLogProbsPolicy):
  """Same dummy policy as above except it stores more things in info."""

  def __init__(self):
    info_spec = {"test": tensor_spec.BoundedTensorSpec([1], tf.int64, 0, 1)}
    super(TfDictInfoAndLogProbs, self).__init__(info_spec=info_spec)

  def _distribution(self, time_step, policy_state):
    distribution_step = super(TfDictInfoAndLogProbs, self)._distribution(
        time_step=time_step, policy_state=policy_state)
    return distribution_step._replace(
        info={"test": tf.constant(1, dtype=tf.int64)})


class TfPolicyTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("SoftUpdate", 0.5, False),
      ("SyncVariables", 1.0, True),
  )
  def testUpdate(self, tau, sort_variables_by_name):
    source_policy = TfPolicyHoldsVariables(
        init_var_value=1., var_scope="source")
    target_policy = TfPolicyHoldsVariables(
        init_var_value=0., var_scope="target")

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for var in self.evaluate(target_policy.variables()):
      self.assertAllEqual(var, np.zeros(var.shape))

    update_op = target_policy.update(
        source_policy, tau=tau, sort_variables_by_name=sort_variables_by_name)
    self.evaluate(update_op)
    for var in self.evaluate(target_policy.variables()):
      self.assertAllEqual(var, np.ones(var.shape) * tau)

  def testClipping(self):
    action_spec = (tensor_spec.BoundedTensorSpec([1], tf.float32, 2, 3),
                   tensor_spec.TensorSpec([1], tf.float32),
                   tensor_spec.BoundedTensorSpec([1], tf.int32, 2, 3),
                   tensor_spec.TensorSpec([1], tf.int32))
    time_step_spec = ts.time_step_spec(action_spec)

    policy = TfPassThroughPolicy(time_step_spec, action_spec, clip=True)

    observation = (tf.constant(1, shape=(1,), dtype=tf.float32),
                   tf.constant(1, shape=(1,), dtype=tf.float32),
                   tf.constant(1, shape=(1,), dtype=tf.int32),
                   tf.constant(1, shape=(1,), dtype=tf.int32))
    time_step = ts.restart(observation)

    clipped_action = self.evaluate(policy.action(time_step).action)
    self.assertEqual(2, clipped_action[0])
    self.assertEqual(1, clipped_action[1])
    self.assertEqual(2, clipped_action[2])
    self.assertEqual(1, clipped_action[3])

  def testObservationsContainExtraFields(self):
    action_spec = {
        "inp": tensor_spec.TensorSpec([1], tf.float32)
    }
    time_step_spec = ts.time_step_spec(observation_spec=action_spec)

    policy = TfPassThroughPolicy(time_step_spec, action_spec, clip=True)

    observation = {"inp": tf.constant(1, shape=(1,), dtype=tf.float32),
                   "extra": tf.constant(1, shape=(1,), dtype=tf.int32)}

    time_step = ts.restart(observation)

    action = policy.action(time_step).action
    distribution = policy.distribution(time_step).action
    tf.nest.assert_same_structure(action, action_spec)
    tf.nest.assert_same_structure(distribution, action_spec)
    self.assertEqual(1, self.evaluate(action["inp"]))
    self.assertEqual(1, self.evaluate(distribution["inp"].sample()))

  def testValidateArgsDisabled(self):
    action_spec = {"blah": ()}
    time_step_spec = ts.time_step_spec(observation_spec=None)
    policy = TfPassThroughPolicy(
        time_step_spec, action_spec, validate_args=False, clip=False)
    observation = (tf.constant(1, shape=(1,), dtype=tf.float32),
                   tf.constant(1, shape=(1,), dtype=tf.float32),
                   tf.constant(1, shape=(1,), dtype=tf.int32),
                   tf.constant(1, shape=(1,), dtype=tf.int32))
    time_step = ts.restart(observation)

    action = self.evaluate(policy.action(time_step).action)
    self.assertAllEqual([[1], [1], [1], [1]], action)

  def testMismatchedDtypes(self):
    with self.assertRaisesRegexp(TypeError, ".*dtype that doesn't match.*"):
      policy = TFPolicyMismatchedDtypes()
      observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
      time_step = ts.restart(observation)
      policy.action(time_step)

  def testMatchedDtypes(self):
    policy = TFPolicyMismatchedDtypes()

    # Overwrite the action_spec to match the dtype of _action.
    policy._action_spec = tensor_spec.BoundedTensorSpec([1], tf.int64, 0, 1)

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observation)
    policy.action(time_step)

  def testMismatchedDtypesListAction(self):
    with self.assertRaisesRegexp(TypeError, ".*dtype that doesn't match.*"):
      policy = TFPolicyMismatchedDtypesListAction()
      observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
      time_step = ts.restart(observation)
      policy.action(time_step)

  def testMatchedDtypesListAction(self):
    policy = TFPolicyMismatchedDtypesListAction()

    # Overwrite the action_spec to match the dtype of _action.
    policy._action_spec = [
        tensor_spec.BoundedTensorSpec([1], tf.int64, 0, 1),
        tensor_spec.BoundedTensorSpec([1], tf.int64, 0, 1)
    ]

    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observation)
    policy.action(time_step)

  def testEmitLogProbability(self):
    policy = TfEmitLogProbsPolicy()
    observation = tf.constant(2., shape=(2, 2), dtype=tf.float32)
    time_step = ts.restart(observation)

    step = self.evaluate(policy.action(time_step))
    self.assertAlmostEqual(step.info.log_probability, np.log(0.2))

  def testKeepInfoAndEmitLogProbability(self):
    policy = TfDictInfoAndLogProbs()
    observation = tf.constant(2., shape=(2, 2), dtype=tf.float32)
    time_step = ts.restart(observation)

    step = self.evaluate(policy.action(time_step))
    self.assertEqual(step.info.get("test", None), 1)
    self.assertAlmostEqual(step.info["log_probability"], np.log(0.2))

  def testAutomaticReset(self):
    observation_spec = tensor_spec.TensorSpec([1], tf.float32)
    action_spec = tensor_spec.TensorSpec([1], tf.float32)
    policy_state_spec = tensor_spec.TensorSpec([1], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)

    policy = TfPassThroughPolicy(
        time_step_spec,
        action_spec,
        policy_state_spec=policy_state_spec,
        automatic_state_reset=True)

    observation = tf.constant(1, dtype=tf.float32, shape=(1, 1))
    reward = tf.constant(1, dtype=tf.float32, shape=(1,))
    time_step = tf.nest.map_structure(lambda *t: tf.concat(t, axis=0),
                                      ts.restart(observation, batch_size=1),
                                      ts.transition(observation, reward),
                                      ts.termination(observation, reward))

    state = self.evaluate(
        policy.action(time_step,
                      policy_state=policy.get_initial_state(3) + 1).state)

    self.assertEqual(0, state[0])
    self.assertEqual(1, state[1])
    self.assertEqual(1, state[2])

    state = self.evaluate(
        policy.distribution(
            time_step, policy_state=policy.get_initial_state(3) + 1).state)

    self.assertEqual(0, state[0])
    self.assertEqual(1, state[1])
    self.assertEqual(1, state[2])

  def testStateShape(self):
    time_step_spec = ts.time_step_spec(tensor_spec.TensorSpec([1], tf.float32))
    action_spec = tensor_spec.TensorSpec([1], tf.float32)
    policy_state_spec = {"foo": tensor_spec.TensorSpec([1], tf.float32),
                         "bar": tensor_spec.TensorSpec([2, 2], tf.int8)}

    policy = TfPassThroughPolicy(
        time_step_spec,
        action_spec,
        policy_state_spec=policy_state_spec)

    # Test state shape with explicit batch_size
    initial_state = policy.get_initial_state(3)
    tf.nest.assert_same_structure(policy_state_spec, initial_state)
    self.assertEqual([3, 1], initial_state["foo"].shape.as_list())
    self.assertEqual([3, 2, 2], initial_state["bar"].shape.as_list())

    # Test state shape with batch_size None
    initial_state = policy.get_initial_state(None)
    tf.nest.assert_same_structure(policy_state_spec, initial_state)
    self.assertEqual([1], initial_state["foo"].shape.as_list())
    self.assertEqual([2, 2], initial_state["bar"].shape.as_list())


if __name__ == "__main__":
  tf.test.main()
