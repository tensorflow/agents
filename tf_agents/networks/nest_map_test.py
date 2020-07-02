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

"""Tests for tf_agents.networks.nest_map."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import os

from absl import flags
import tensorflow.compat.v2 as tf

from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.policies import policy_saver
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class MyPolicy(tf_policy.TFPolicy):

  def __init__(self, time_step_spec, net):
    super(MyPolicy, self).__init__(
        time_step_spec,
        action_spec=tf.TensorSpec((None,), tf.float32))
    self._net = net

  def _action(self, time_step, policy_state=(), seed=None):
    out, _ = self._net(time_step.observation)
    out = tf.math.add(*tf.nest.flatten(out))
    return policy_step.PolicyStep(out, (), ())


class NestFlattenTest(test_utils.TestCase):

  def testNestFlatten(self):
    layer = nest_map.NestFlatten()
    outputs = layer({'a': 1, 'b': 2})
    self.assertEqual(self.evaluate(outputs), [1, 2])


class NestMapTest(test_utils.TestCase):

  def setUp(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x.')
    super(NestMapTest, self).setUp()

  def testCreateAndCall(self):
    net = sequential.Sequential([
        nest_map.NestMap(
            {'inp1': tf.keras.layers.Dense(8),
             'inp2': sequential.Sequential([
                 tf.keras.layers.Conv2D(2, 3),
                 # Convert 3 inner dimensions to [8] for RNN.
                 inner_reshape.InnerReshape([None] * 3, [8]),
             ]),
             'inp3': tf.keras.layers.LSTM(
                 8, return_state=True, return_sequences=True)}),
        nest_map.NestFlatten(),
        tf.keras.layers.Add()])
    self.assertEqual(
        net.state_spec,
        ({
            'inp1': (),
            'inp2': (),
            'inp3': (2 * [tf.TensorSpec(shape=(8,), dtype=tf.float32)],),
        },))
    output_spec = net.create_variables(
        {
            'inp1': tf.TensorSpec(shape=(3,), dtype=tf.float32),
            'inp2': tf.TensorSpec(shape=(4, 4, 2,), dtype=tf.float32),
            'inp3': tf.TensorSpec(shape=(3,), dtype=tf.float32),
        })
    self.assertEqual(output_spec, tf.TensorSpec(shape=(8,), dtype=tf.float32))

    inputs = {
        'inp1': tf.ones((8, 10, 3), dtype=tf.float32),
        'inp2': tf.ones((8, 10, 4, 4, 2), dtype=tf.float32),
        'inp3': tf.ones((8, 10, 3), dtype=tf.float32)
    }
    output, next_state = net(inputs)
    self.assertEqual(output.shape, tf.TensorShape([8, 10, 8]))
    self.assertEqual(
        tf.nest.map_structure(lambda t: t.shape, next_state),
        ({
            'inp1': (),
            'inp2': (),
            'inp3': (2 * [tf.TensorShape([8, 8])],),
        },))

    # Test passing in a state.
    output, next_state = net(inputs, next_state)
    self.assertEqual(output.shape, tf.TensorShape([8, 10, 8]))
    self.assertEqual(
        tf.nest.map_structure(lambda t: t.shape, next_state),
        ({
            'inp1': (),
            'inp2': (),
            'inp3': (2 * [tf.TensorShape([8, 8])],),
        },))

  def testIncompatibleStructureInputs(self):
    with self.assertRaisesRegex(
        ValueError,
        r'`nested_layers` and `input_spec` do not have matching structures'):
      nest_map.NestMap(
          tf.keras.layers.Dense(8),
          input_spec={'ick': tf.TensorSpec(8, tf.float32)})

    with self.assertRaisesRegex(
        ValueError,
        r'`inputs` and `self.nested_layers` do not have matching structures'):
      net = nest_map.NestMap(tf.keras.layers.Dense(8))
      net.create_variables({'ick': tf.TensorSpec((1,), dtype=tf.float32)})

    with self.assertRaisesRegex(
        ValueError,
        r'`inputs` and `self.nested_layers` do not have matching structures'):
      net = nest_map.NestMap(tf.keras.layers.Dense(8))
      net({'ick': tf.constant([[1.0]])})

    with self.assertRaisesRegex(
        ValueError,
        r'`network_state` and `state_spec` do not have matching structures'):
      net = nest_map.NestMap(
          tf.keras.layers.LSTM(8, return_state=True, return_sequences=True))
      net(tf.ones((1, 2)), network_state=(tf.ones((1, 1)), ()))

  def testPolicySaverCompatibility(self):
    observation_spec = {
        'a': tf.TensorSpec(4, tf.float32),
        'b': tf.TensorSpec(3, tf.float32)
    }
    time_step_tensor_spec = ts.time_step_spec(observation_spec)
    net = nest_map.NestMap(
        {'a': tf.keras.layers.LSTM(8, return_state=True, return_sequences=True),
         'b': tf.keras.layers.Dense(8)})
    net.create_variables(observation_spec)
    policy = MyPolicy(time_step_tensor_spec, net)

    sample = tensor_spec.sample_spec_nest(
        time_step_tensor_spec, outer_dims=(5,))

    step = policy.action(sample)
    self.assertEqual(step.action.shape.as_list(), [5, 8])

    train_step = common.create_variable('train_step')
    saver = policy_saver.PolicySaver(policy, train_step=train_step)
    self.initialize_v1_variables()

    with self.cached_session():
      saver.save(os.path.join(FLAGS.test_tmpdir, 'nest_map_model'))


if __name__ == '__main__':
  test_utils.main()
