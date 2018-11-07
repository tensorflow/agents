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

"""Test for tf_agents.utils.eager_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
import tensorflow as tf

from tf_agents.utils import eager_utils
from tensorflow.python.eager import context  # TF internal
from tensorflow.python.framework import test_util  # TF internal
from tensorflow.python.keras.engine import network as keras_network  # TF internal


def input_fn():
  tf.set_random_seed(1)
  inputs = tf.constant([[1, 2], [2, 3], [3, 4]], dtype=tf.float32)
  labels = tf.constant([[0], [1], [2]])
  return inputs, labels


class Network(keras_network.Network):

  def __init__(self, name=None):
    super(Network, self).__init__(name=name)
    self._layer = tf.keras.layers.Dense(
        3,
        kernel_initializer=tf.ones_initializer(),
        name='logits')

  def call(self, inputs):
    return self._layer(inputs)


class Model(object):

  def __init__(self, name, network):
    self._name = name
    self._network = network

  def __call__(self, inputs):
    return self._network(inputs)

  @property
  def variables(self):
    return self._network.variables

  @property
  def trainable_variables(self):
    return self._network.trainable_variables

  @eager_utils.future_in_eager_mode
  def loss_fn(self, inputs, labels):
    logits = self._network(inputs)
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)


@eager_utils.future_in_eager_mode
def minimize_loss(loss, optimizer):
  return optimizer.minimize(loss)


class Aux(object):

  def __init__(self):
    pass

  def method(self, inputs, labels, param=0):
    assert isinstance(self, Aux), self
    return inputs, labels, tf.convert_to_tensor(param)


def aux_function(inputs, labels, param=0):
  return inputs, labels, tf.convert_to_tensor(param)


@parameterized.named_parameters(
    ('.func_eager', aux_function, context.eager_mode),
    ('.func_graph', aux_function, context.graph_mode),
    ('.method_eager', Aux().method, context.eager_mode),
    ('.method_graph', Aux().method, context.graph_mode),
)
class FutureTest(tf.test.TestCase, parameterized.TestCase):

  def testCreate(self, func_or_method, run_mode):
    with run_mode():
      future = eager_utils.Future(input_fn)
      self.assertTrue(callable(future))
      self.assertIsInstance(future, eager_utils.Future)
      inputs, labels = future()
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])

  def testArgsAtInit(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, inputs, labels)
      inputs, labels, param = future()
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)

  def testArgsAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method)
      inputs, labels, param = future(inputs, labels)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)

  def testArgsAtCallOverwriteKwargsInit(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, param=1)
      inputs, labels, param = future(inputs, labels, 0)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)

  def testKWArgsAtInit(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(
          func_or_method, inputs=inputs, labels=labels, param=1)
      inputs, labels, param = future()
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 1)

  def testKWArgsAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method)
      inputs, labels, param = future(inputs=inputs, labels=labels, param=1)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 1)

  def testArgsAtInitKWArgsAtInit(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, inputs, labels=labels)
      inputs, labels, param = future()
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)

  def testArgsAtInitKWArgsAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, inputs, param=1)
      inputs, labels, param = future(labels=labels)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 1)

  def testOverwriteKWArgsAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, param=-1)
      inputs, labels, param = future(inputs, labels, param=1)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 1)

  def testArgsatInitOverwritedKWArgsAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, inputs, param=-1)
      inputs, labels, param = future(labels=labels, param=1)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 1)

  def testPartialArgsAtCallRaisesError(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, inputs)
      with self.assertRaisesRegexp(TypeError, 'argument'):
        future(labels)

  def testArgsAtInitArgsReplacedAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, labels, inputs)
      inputs, labels, param = future(inputs, labels)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)

  def testArgsAtCallKWArgsAtInit(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method, labels=labels)
      inputs, labels, param = future(inputs)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)

  def testArgsAtCallKWArgsAtCall(self, func_or_method, run_mode):
    with run_mode():
      inputs, labels = input_fn()
      future = eager_utils.Future(func_or_method)
      inputs, labels, param = future(inputs, labels=labels)
      self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
      self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])
      self.assertEqual(self.evaluate(param), 0)


class FutureInEagerModeTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testCreate(self):
    decorator = eager_utils.future_in_eager_mode(input_fn)
    self.assertTrue(callable(decorator))
    if context.executing_eagerly():
      self.assertTrue(isinstance(decorator(), eager_utils.Future))
      inputs, labels = decorator()()
    else:
      inputs, labels = decorator()
    self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
    self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])

  def testDecorator(self):
    @eager_utils.future_in_eager_mode
    def aux_fn(inputs, labels):
      return inputs, labels

    self.assertTrue(callable(aux_fn))
    inputs, labels = input_fn()
    outputs = aux_fn(inputs, labels)

    if context.executing_eagerly():
      self.assertTrue(isinstance(outputs, eager_utils.Future))
      inputs, labels = outputs.__call__()
    else:
      inputs, labels = outputs
    self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
    self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])

  def testDelayedArgs(self):
    @eager_utils.future_in_eager_mode
    def aux_fn(inputs, labels):
      return inputs, labels

    self.assertTrue(callable(aux_fn))
    inputs, labels = input_fn()
    outputs = aux_fn(inputs, labels)

    if context.executing_eagerly():
      self.assertTrue(isinstance(outputs, eager_utils.Future))
      inputs, labels = outputs.__call__()
    else:
      inputs, labels = outputs
    self.assertAllEqual(self.evaluate(inputs), [[1, 2], [2, 3], [3, 4]])
    self.assertAllEqual(self.evaluate(labels), [[0], [1], [2]])


class EagerUtilsTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testModel(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    expected_loss = 1.098612
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  @test_util.run_in_graph_and_eager_modes()
  def testLossDecreasesAfterTrainStep(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step = minimize_loss(loss, optimizer)
    initial_loss = 1.098612
    final_loss = 1.064379
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), initial_loss)
    self.evaluate(train_step)
    self.assertAllClose(self.evaluate(loss), final_loss)


class CreateTrainOpTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testLossDecreasesAfterTrainOp(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step = eager_utils.create_train_step(loss, optimizer)
    initial_loss = 1.098612
    final_loss = 1.064379
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(train_step), initial_loss)
    self.assertAllClose(self.evaluate(train_step), final_loss)

  @test_util.run_in_graph_and_eager_modes()
  def testCreateTrainOpWithTotalLossFn(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    model_2 = Model('model_2', Network())
    loss_2 = model_2.loss_fn(inputs, labels)

    @eager_utils.future_in_eager_mode
    def tuple_loss(loss, loss_2):
      return (loss() if callable(loss) else loss,
              loss_2() if callable(loss_2) else loss_2)
    tuple_loss_value = tuple_loss(loss, loss_2)

    def first_element(tuple_value):
      return tuple_value[0]

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    loss = eager_utils.create_train_step(
        tuple_loss_value, optimizer, total_loss_fn=first_element)
    initial_loss = 1.098612
    final_loss = 1.064379
    self.evaluate(tf.global_variables_initializer())
    train_step_model_0, train_step_model_1 = self.evaluate(loss)
    self.assertAllClose(train_step_model_0, initial_loss)
    self.assertAllClose(train_step_model_1, initial_loss)
    train_step_model_0, train_step_model_1 = self.evaluate(loss)
    self.assertAllClose(train_step_model_0, final_loss)
    # model_1 was not updated since its loss is not being optimized: only
    # the first element output was optimized.
    self.assertAllClose(train_step_model_1, initial_loss)

  @test_util.run_in_graph_and_eager_modes()
  def testMultipleCallsTrainStep(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step = eager_utils.create_train_step(loss, optimizer)
    initial_loss = 1.098612
    final_loss = 1.033917
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(train_step), initial_loss)
    if context.executing_eagerly():
      for _ in range(5):
        self.evaluate(train_step(inputs, labels))
      self.assertAllClose(self.evaluate(train_step(inputs, labels)), final_loss)
    else:
      for _ in range(5):
        self.evaluate(train_step)
      self.assertAllClose(self.evaluate(train_step), final_loss)

  @test_util.run_in_graph_and_eager_modes()
  def testVariablesToTrain(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    if context.executing_eagerly():
      variables_to_train = lambda: model.trainable_variables
    else:
      model(inputs)
      variables_to_train = model.trainable_variables
      self.assertEqual(len(variables_to_train), 2)
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step = eager_utils.create_train_step(
        loss, optimizer, variables_to_train=variables_to_train)
    initial_loss = 1.098612
    final_loss = 1.064379
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(train_step), initial_loss)
    self.assertAllClose(self.evaluate(train_step), final_loss)
    self.assertEqual(len(model.trainable_variables), 2)


class HasSelfClsArgTest(tf.test.TestCase):

  def testDirect(self):

    def func():
      pass

    func2 = lambda: 0

    class A(object):

      def method(self):
        pass

      @classmethod
      def class_method(cls):
        pass

      @staticmethod
      def static_method():
        pass

    self.assertFalse(eager_utils.has_self_cls_arg(func))
    self.assertFalse(eager_utils.has_self_cls_arg(func2))
    self.assertFalse(eager_utils.has_self_cls_arg(A.static_method))

    self.assertTrue(eager_utils.has_self_cls_arg(A.method))
    self.assertTrue(eager_utils.has_self_cls_arg(A().method))
    self.assertTrue(eager_utils.has_self_cls_arg(A.class_method))
    self.assertTrue(eager_utils.has_self_cls_arg(A().class_method))

    self.assertTrue(eager_utils.has_self_cls_arg(A.__dict__['method']))
    self.assertTrue(eager_utils.has_self_cls_arg(A.__dict__['class_method']))
    self.assertFalse(eager_utils.has_self_cls_arg(A.__dict__['static_method']))

  def testDecorator(self):

    def decorator(method):

      @functools.wraps(method)
      def _decorator(*args, **kwargs):
        method(*args, **kwargs)

      return _decorator

    class A(object):

      @decorator
      def method(self):
        pass

      @staticmethod
      @decorator
      def static_method():
        pass

      @classmethod
      @decorator
      def class_method(cls):
        pass

    self.assertTrue(eager_utils.has_self_cls_arg(A.method))
    self.assertTrue(eager_utils.has_self_cls_arg(A.class_method))
    self.assertFalse(eager_utils.has_self_cls_arg(A.static_method))


if __name__ == '__main__':
  tf.test.main()
