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

"""Tests for tf_agents.networks.network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents import specs
from tf_agents.distributions import utils as distribution_utils
from tf_agents.keras_layers import rnn_wrapper
from tf_agents.networks import network
from tf_agents.utils import common


tfd = tfp.distributions


class BaseNetwork(network.Network):

  # pylint: disable=useless-super-delegation
  def __init__(self, v1, **kwargs):
    super(BaseNetwork, self).__init__(v1, **kwargs)
  # pylint: enable=useless-super-delegation


class NetworkNoExtraKeywordsInCallSignature(network.Network):

  def call(self, inputs):
    return inputs, ()


class MockNetwork(BaseNetwork):

  def __init__(self, param1, param2, kwarg1=2, kwarg2=3):
    self.param1 = param1
    self.param2 = param2
    self.kwarg1 = kwarg1
    self.kwarg2 = kwarg2
    super(MockNetwork, self).__init__(param1,
                                      state_spec=(),
                                      name='mock')

  def build(self, *args, **kwargs):
    self.var1 = common.create_variable(
        'variable', dtype=tf.float32, trainable=False)
    self.var2 = common.create_variable(
        'trainable_variable', dtype=tf.float32, trainable=True)

  def call(self, observations, step_type, network_state=None):
    return self.var1 + self.var2 + observations, ()


class NoInitNetwork(MockNetwork):
  pass


class GnarlyNetwork(network.Network):

  def __init__(self):
    k1 = tf.keras.Sequential([
        tf.keras.layers.Dense(
            32,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization()
    ], name='a')
    k2 = tf.keras.layers.Dense(12, name='b')
    super(GnarlyNetwork, self).__init__(
        input_tensor_spec=tf.TensorSpec(dtype=tf.float32, shape=(2,)),
        state_spec=(), name=None)
    self._k1 = k1
    self._k2 = k2

  def call(self, observations, step_type, network_state=None):
    return self._k2(self._k1(observations)), network_state


class NetworkTest(tf.test.TestCase):

  def test_copy_works(self):
    network1 = MockNetwork((), 1)
    network2 = network1.copy()

    self.assertNotEqual(network1, network2)
    self.assertEqual((), network2.param1)
    self.assertEqual(1, network2.param2)
    self.assertEqual(2, network2.kwarg1)
    self.assertEqual(3, network2.kwarg2)

  def test_noinit_copy_works(self):
    network1 = NoInitNetwork((), 1)
    network2 = network1.copy()

    self.assertNotEqual(network1, network2)
    self.assertEqual((), network2.param1)
    self.assertEqual(1, network2.param2)
    self.assertEqual(2, network2.kwarg1)
    self.assertEqual(3, network2.kwarg2)

  def test_too_many_args_raises_appropriate_error(self):
    with self.assertRaisesRegexp(TypeError, '__init__.*given'):
      # pylint: disable=too-many-function-args
      MockNetwork(0, 1, 2, 3, 4, 5, 6)  # pytype: disable=wrong-arg-count

  def test_assert_input_spec(self):
    spec = specs.TensorSpec([], tf.int32, 'action')
    net = MockNetwork(spec, 1)
    with self.assertRaises(ValueError):
      net((1, 2), 2)

  def test_create_variables(self):
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([2], tf.float32, 'action')
    net = MockNetwork(observation_spec, action_spec)
    self.assertFalse(net.built)
    with self.assertRaises(ValueError):
      net.variables  # pylint: disable=pointless-statement
    output_spec = net.create_variables()
    # MockNetwork adds some variables to observation, which has shape [bs, 1]
    self.assertEqual(output_spec, tf.TensorSpec([1], dtype=tf.float32))
    self.assertTrue(net.built)
    self.assertLen(net.variables, 2)
    self.assertLen(net.trainable_variables, 1)

  def test_create_variables_distribution(self):
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([2], tf.float32, 'action')
    net = MockNetwork(observation_spec, action_spec)
    self.assertFalse(net.built)
    with self.assertRaises(ValueError):
      net.variables  # pylint: disable=pointless-statement
    output_spec = net.create_variables()
    # MockNetwork adds some variables to observation, which has shape [bs, 1]
    self.assertEqual(output_spec, tf.TensorSpec([1], dtype=tf.float32))
    self.assertTrue(net.built)
    self.assertLen(net.variables, 2)
    self.assertLen(net.trainable_variables, 1)

  def test_summary_no_exception(self):
    """Tests that Network.summary() does not throw an exception."""
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([2], tf.float32, 'action')
    net = MockNetwork(observation_spec, action_spec)
    net.create_variables()
    net.summary()

  def test_access_deep_layers_weights_and_losses(self):
    net = GnarlyNetwork()
    net.create_variables(training=True)
    layer_names = sorted([l.name for l in net.layers])
    losses = net.losses
    trainable_weight_names = sorted([w.name for w in net.trainable_weights])
    non_trainable_weight_names = sorted(
        [w.name for w in net.non_trainable_weights])
    self.assertEqual(layer_names, ['a', 'b'])
    self.assertLen(losses, 2)
    for loss in losses:
      self.assertEqual(loss.dtype, tf.float32)
      self.assertEqual(loss.shape, ())
    self.assertEqual(
        [x.lstrip('gnarly_network/') for x in trainable_weight_names],
        ['batch_normalization/beta:0',
         'batch_normalization/gamma:0',
         'dense/bias:0',
         'dense/kernel:0',
         'dense_1/bias:0',
         'dense_1/kernel:0',
         'b/bias:0',
         'b/kernel:0'])
    self.assertEqual(
        [x.lstrip('gnarly_network/') for x in non_trainable_weight_names],
        ['batch_normalization/moving_mean:0',
         'batch_normalization/moving_variance:0'])

  def test_dont_complain_if_no_network_state_in_call_signature(self):
    net = NetworkNoExtraKeywordsInCallSignature()
    out, _ = net(1, network_state=None)  # This shouldn't complain.
    self.assertAllEqual(out, 1)
    out, _ = net(1, step_type=3, network_state=None)  # This shouldn't complain.
    self.assertAllEqual(out, 1)


class CreateVariablesTest(parameterized.TestCase, tf.test.TestCase):

  def testNetworkCreate(self):
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([2], tf.float32, 'action')
    net = MockNetwork(observation_spec, action_spec)
    self.assertFalse(net.built)
    with self.assertRaises(ValueError):
      net.variables  # pylint: disable=pointless-statement
    output_spec = network.create_variables(net)
    # MockNetwork adds some variables to observation, which has shape [bs, 1]
    self.assertEqual(output_spec, tf.TensorSpec([1], dtype=tf.float32))
    self.assertTrue(net.built)
    self.assertLen(net.variables, 2)
    self.assertLen(net.trainable_variables, 1)

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      (
          'Dense',
          lambda: tf.keras.layers.Dense(3),
          tf.TensorSpec((5,), tf.float32),  # input_spec
          tf.TensorSpec((3,), tf.float32),  # expected_output_spec
          (),  # expected_state_spec
      ),
      (
          'LSTMCell',
          lambda: tf.keras.layers.LSTMCell(3),
          tf.TensorSpec((5,), tf.float32),
          tf.TensorSpec((3,), tf.float32),
          [tf.TensorSpec((3,), tf.float32),
           tf.TensorSpec((3,), tf.float32)],
      ),
      (
          'LSTMCellInRNN',
          lambda: rnn_wrapper.RNNWrapper(
              tf.keras.layers.RNN(
                  tf.keras.layers.LSTMCell(3),
                  return_state=True,
                  return_sequences=True)
          ),
          tf.TensorSpec((5,), tf.float32),
          tf.TensorSpec((3,), tf.float32),
          [tf.TensorSpec((3,), tf.float32),
           tf.TensorSpec((3,), tf.float32)],
      ),
      (
          'LSTM',
          lambda: rnn_wrapper.RNNWrapper(
              tf.keras.layers.LSTM(
                  3,
                  return_state=True,
                  return_sequences=True)
          ),
          tf.TensorSpec((5,), tf.float32),
          tf.TensorSpec((3,), tf.float32),
          [tf.TensorSpec((3,), tf.float32),
           tf.TensorSpec((3,), tf.float32)],
      ),
      (
          'TimeDistributed',
          lambda: tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3)),
          tf.TensorSpec((5,), tf.float32),
          tf.TensorSpec((3,), tf.float32),
          ()
      ),
      (
          'Conv2D',
          lambda: tf.keras.layers.Conv2D(2, 3),
          tf.TensorSpec((28, 28, 5), tf.float32),
          tf.TensorSpec((26, 26, 2), tf.float32),
          ()
      ),
      (
          'SequentialOfDense',
          lambda: tf.keras.Sequential([tf.keras.layers.Dense(3)] * 2),
          tf.TensorSpec((5,), tf.float32),
          tf.TensorSpec((3,), tf.float32),
          ()
      ),
      (
          'NormalDistribution',
          lambda: tf.keras.Sequential(
              [tf.keras.layers.Dense(3),
               tf.keras.layers.Lambda(
                   lambda x: tfd.Normal(loc=x, scale=x**2))]),
          tf.TensorSpec((5,), tf.float32),
          distribution_utils.DistributionSpecV2(
              event_shape=tf.TensorShape(()),
              dtype=tf.float32,
              parameters=distribution_utils.Params(
                  type_=tfd.Normal,
                  params=dict(
                      loc=tf.TensorSpec((3,), tf.float32),
                      scale=tf.TensorSpec((3,), tf.float32),
                  ))),
          ()
      ),
  )
  # pylint: enable=g-long-λ
  def testKerasLayerCreate(self, layer_fn, input_spec, expected_output_spec,
                           expected_state_spec):
    layer = layer_fn()
    with self.assertRaisesRegex(ValueError, 'an input_spec is required'):
      network.create_variables(layer)
    output_spec = network.create_variables(layer, input_spec)
    self.assertTrue(layer.built)
    self.assertEqual(
        output_spec, expected_output_spec,
        '\n{}\nvs.\n{}\n'.format(output_spec, expected_output_spec))
    output_spec_2 = network.create_variables(layer, input_spec)
    self.assertEqual(output_spec_2, expected_output_spec)
    state_spec = getattr(layer, '_network_state_spec', None)
    self.assertEqual(state_spec, expected_state_spec)


class MockStateFullNetwork(BaseNetwork):

  def __init__(self, input_spec, state_spec):
    super(MockStateFullNetwork, self).__init__(input_spec,
                                               state_spec=state_spec,
                                               name='statefullmock')

  def build(self, *args, **kwargs):
    self.var = common.create_variable(
        'trainable_variable', dtype=tf.float32, trainable=True)
    self.state = common.create_variable(
        'state', dtype=tf.float32, trainable=False)

  def call(self, observations, network_state=None):
    return self.var + observations, self.state + network_state


class StateFullNetworkTest(tf.test.TestCase):

  def test_specs(self):
    input_spec = tf.TensorSpec([], tf.float32, 'inputs')
    state_spec = tf.TensorSpec([], tf.float32, 'state')
    net = MockStateFullNetwork(input_spec, state_spec)
    self.assertEqual(input_spec, net.input_tensor_spec)
    self.assertEqual(state_spec, net.state_spec)

  def test_empty_state(self):
    input_spec = tf.TensorSpec([], tf.float32, 'inputs')
    net = MockStateFullNetwork(input_spec, ())
    self.assertEqual(input_spec, net.input_tensor_spec)
    self.assertEqual((), net.state_spec)
    net.create_variables()

  def test_wrong_new_state(self):
    input_spec = tf.TensorSpec([], tf.float32, 'inputs')
    net = MockStateFullNetwork(input_spec, ((), ()))
    self.assertEqual(input_spec, net.input_tensor_spec)
    self.assertEqual(((), ()), net.state_spec)
    with self.assertRaises(ValueError):
      net.create_variables()

  def test_copy_works(self):
    input_spec = tf.TensorSpec([], tf.float32, 'inputs')
    state_spec = tf.TensorSpec([], tf.float32, 'state')
    network1 = MockStateFullNetwork(input_spec, state_spec)
    network2 = network1.copy()

    self.assertNotEqual(network1, network2)
    self.assertEqual(network1.input_tensor_spec, network2.input_tensor_spec)
    self.assertEqual(network1.state_spec, network2.state_spec)

  def test_create_variables(self):
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([1], tf.float32, 'action')
    input_spec = (observation_spec, action_spec)
    state_spec = tf.TensorSpec([], tf.float32, 'state')
    net = MockStateFullNetwork(input_spec, state_spec)
    self.assertFalse(net.built)
    with self.assertRaises(ValueError):
      net.variables  # pylint: disable=pointless-statement
    output_spec = net.create_variables()

    self.assertEqual(output_spec, tf.TensorSpec([1, 1], dtype=tf.float32))
    self.assertTrue(net.built)
    self.assertLen(net.variables, 2)
    self.assertLen(net.trainable_variables, 1)

  def test_call(self):
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    state_spec = tf.TensorSpec([], tf.float32, 'state')
    net = MockStateFullNetwork(observation_spec, state_spec)

    initial_state = net._get_initial_state(batch_size=1)
    observation = tf.constant([1.0])
    outputs, new_state = net(observation, initial_state)
    # Only needed for TF1
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertEqual(self.evaluate(outputs), 1.0)
    self.assertEqual(self.evaluate(new_state), 0.0)

if __name__ == '__main__':
  tf.test.main()
