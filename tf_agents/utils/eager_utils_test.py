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

"""Test for tf_agents.utils.eager_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import eager_utils
from tf_agents.utils import test_utils

from tensorflow.python.eager import context  # TF internal
from tensorflow.python.framework import test_util  # TF internal


def input_fn():
  tf.compat.v1.set_random_seed(1)
  inputs = tf.constant([[1, 2], [2, 3], [3, 4]], dtype=tf.float32)
  labels = tf.constant([[0], [1], [2]])
  return inputs, labels


class Network(tf.keras.layers.Layer):

  def __init__(self, name=None):
    super(Network, self).__init__(name=name)
    self._layer = tf.keras.layers.Dense(
        3, kernel_initializer=tf.keras.initializers.Ones(), name='logits')

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
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)


@eager_utils.future_in_eager_mode
def minimize_loss(loss, optimizer):
  return optimizer.minimize(loss)


class Aux(object):

  def __init__(self):
    pass

  def method(self, inputs, labels, param=0):
    assert isinstance(self, Aux), self
    return inputs, labels, tf.convert_to_tensor(value=param)


def aux_function(inputs, labels, param=0):
  return inputs, labels, tf.convert_to_tensor(value=param)


@parameterized.named_parameters(
    ('.func_eager', aux_function, context.eager_mode),
    ('.func_graph', aux_function, context.graph_mode),
    ('.method_eager', Aux().method, context.eager_mode),
    ('.method_graph', Aux().method, context.graph_mode),
)
class FutureTest(test_utils.TestCase, parameterized.TestCase):

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


class FutureInEagerModeTest(test_utils.TestCase):

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


class EagerUtilsTest(test_utils.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testModel(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    expected_loss = 1.098612
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), expected_loss)

  @test_util.run_in_graph_and_eager_modes()
  def testLossDecreasesAfterTrainStep(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train_step = minimize_loss(loss, optimizer)
    initial_loss = 1.098612
    final_loss = 1.064379
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(loss), initial_loss)
    self.evaluate(train_step)
    self.assertAllClose(self.evaluate(loss), final_loss)


class ClipGradsTest(test_utils.TestCase):

  def testClipGrads(self):
    xs = tf.Variable(0.0)
    grads = tf.constant(4.0)
    gradients_to_variables = [(grads, xs)]
    clipped_gradients_to_variables = eager_utils.clip_gradient_norms(
        gradients_to_variables, 3.0)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAlmostEqual(4.0, self.evaluate(gradients_to_variables[0][0]))
    self.assertAlmostEqual(3.0,
                           self.evaluate(clipped_gradients_to_variables[0][0]))

  def testClipGradsIndexedSlices(self):
    xs = tf.Variable(0.0)
    grads = tf.IndexedSlices(values=tf.constant(4.0),
                             indices=tf.constant([0]),
                             dense_shape=None)
    gradients_to_variables = [(grads, xs)]
    clipped_gradients_to_variables = eager_utils.clip_gradient_norms(
        gradients_to_variables, 3.0)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAlmostEqual(
        4.0, self.evaluate(gradients_to_variables[0][0].values))
    self.assertAlmostEqual(
        3.0, self.evaluate(clipped_gradients_to_variables[0][0].values))

  def testClipGradsFn(self):
    xs = tf.Variable(0.0)
    grads = tf.constant(4.0)
    gradients_to_variables = [(grads, xs)]
    clipped_gradients_to_variables = eager_utils.clip_gradient_norms_fn(3.0)(
        gradients_to_variables)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAlmostEqual(4.0, self.evaluate(gradients_to_variables[0][0]))
    self.assertAlmostEqual(3.0,
                           self.evaluate(clipped_gradients_to_variables[0][0]))


class CreateTrainOpTest(test_utils.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testLossDecreasesAfterTrainOp(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train_step = eager_utils.create_train_step(loss, optimizer)
    initial_loss = 1.098612
    final_loss = 1.064379
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(train_step), initial_loss)
    self.assertAllClose(self.evaluate(loss), final_loss)

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

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    loss = eager_utils.create_train_step(
        tuple_loss_value, optimizer, total_loss_fn=first_element)
    expected_loss = 1.098612
    self.evaluate(tf.compat.v1.global_variables_initializer())
    train_step_model_0, train_step_model_1 = self.evaluate(loss)
    self.assertAllClose(train_step_model_0, expected_loss)
    self.assertAllClose(train_step_model_1, expected_loss)

  @test_util.run_in_graph_and_eager_modes()
  def testMultipleCallsTrainStep(self):
    inputs, labels = input_fn()
    model = Model('model', Network())
    loss = model.loss_fn(inputs, labels)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train_step = eager_utils.create_train_step(loss, optimizer)
    initial_loss = 1.098612
    final_loss = 1.033917
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(train_step), initial_loss)
    if context.executing_eagerly():
      for _ in range(5):
        train_step = eager_utils.create_train_step(loss, optimizer)
      train_step = eager_utils.create_train_step(loss, optimizer)
      self.assertAllClose(self.evaluate(train_step), final_loss)
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
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train_step = eager_utils.create_train_step(
        loss, optimizer, variables_to_train=variables_to_train)
    expected_loss = 1.098612
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(self.evaluate(train_step), expected_loss)
    self.assertEqual(len(model.trainable_variables), 2)


class HasSelfClsArgTest(test_utils.TestCase):

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


@eager_utils.np_function
def meshgrid(low, high, nx=2, ny=3):
  x = np.linspace(low, high, nx)
  y = np.linspace(low, high, ny)
  return np.meshgrid(x, y)


@eager_utils.np_function(output_dtypes=np.float32)
def mean(x):
  return np.mean(x)


@eager_utils.np_function(output_dtypes=lambda x: (x, x))
def repeat(x):
  return x, x


@eager_utils.np_function(output_dtypes=lambda x, y: {'x': x, 'y': y})
def dictionary(x, y):
  return {'x': x, 'y': y}


class NpFunctionTest(test_utils.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testMeshGrid(self):
    xv, yv = meshgrid(tf.constant(0), tf.constant(1))
    self.assertAllEqual(self.evaluate(xv), [[0., 1.], [0., 1.], [0., 1.]])
    self.assertAllEqual(self.evaluate(yv), [[0., 0.], [.5, .5], [1., 1.]])
    xv, yv = meshgrid(tf.constant(0.), tf.constant(1.))
    self.assertAllEqual(self.evaluate(xv), [[0., 1.], [0., 1.], [0., 1.]])
    self.assertAllEqual(self.evaluate(yv), [[0., 0.], [.5, .5], [1., 1.]])

  @test_util.run_in_graph_and_eager_modes()
  def testMeshGridKwargs(self):
    xv, yv = meshgrid(tf.constant(0), tf.constant(1), nx=2, ny=2)
    self.assertAllEqual(self.evaluate(xv), [[0., 1.], [0., 1.]])
    self.assertAllEqual(self.evaluate(yv), [[0., 0.], [1., 1.]])

  @test_util.run_in_graph_and_eager_modes()
  def testVariables(self):
    a, b = tf.Variable(0), tf.Variable(1)
    xv, yv = meshgrid(a, b, nx=2, ny=2)
    self.evaluate(tf.compat.v1.initializers.global_variables())
    self.assertAllEqual(self.evaluate(xv), [[0., 1.], [0., 1.]])
    self.assertAllEqual(self.evaluate(yv), [[0., 0.], [1., 1.]])

  @test_util.run_in_graph_and_eager_modes()
  def testGetOutputDtypesInts2Floats(self):
    x = tf.constant([1, 2, 3])
    mean_x = mean(x)
    self.assertEqual(self.evaluate(mean_x), 2.)

  @test_util.run_in_graph_and_eager_modes()
  def testGetOutputDtypesFloats2Floats(self):
    x = tf.constant([1., 2., 3.])
    mean_x = mean(x)
    self.assertEqual(self.evaluate(mean_x), 2.)

  @test_util.run_in_graph_and_eager_modes()
  def testIdentityDtypes(self):
    x = tf.constant([1])
    self.assertAllEqual(self.evaluate(repeat(x)), ([1], [1]))
    y = tf.constant([1.])
    self.assertAllEqual(self.evaluate(repeat(y)), ([1.], [1.]))

  @test_util.run_in_graph_and_eager_modes()
  def testInline(self):
    square = eager_utils.np_function(np.square)
    x = tf.constant([1, 2, 3])
    self.assertAllEqual([1, 4, 9], self.evaluate(square(x)))
    y = tf.constant([1., 2., 3.])
    self.assertAllEqual([1., 4., 9.], self.evaluate(square(y)))

  @test_util.run_in_graph_and_eager_modes()
  def testOutputDictionary(self):
    x = tf.constant([1])
    y = tf.constant([1.])
    outputs = dictionary(x, y)
    self.assertAllEqual([1], self.evaluate(outputs['x']))
    self.assertAllEqual([1.], self.evaluate(outputs['y']))


@eager_utils.np_function(output_dtypes=np.float32)
def np_descent(x, d, mu, n_epochs):
  n = len(x)
  f = 2 / n

  y = np.zeros(n)
  err = np.zeros(n)
  w = np.zeros(2)
  grad = np.zeros(2)

  for _ in itertools.repeat(None, n_epochs):
    np.subtract(d, y, out=err)
    grad[:] = [f * np.sum(err), f * np.dot(err, x)]
    w = w + mu * grad
    y = w[0] + w[1] * x
  return w


class NpDescentTest(test_utils.TestCase):

  def setUp(self):
    np.random.seed(444)
    n = 10000
    sigma = 0.1
    noise = sigma * np.random.randn(n)
    self._x = np.linspace(0, 2, n)
    self._d = 3 + 2 * self._x + noise

  @test_util.run_in_graph_and_eager_modes()
  def testSolve(self):
    x, d = tf.constant(self._x), tf.constant(self._d)
    w = np_descent(x, d, mu=0.001, n_epochs=10000)
    self.assertAllClose([2.96, 2.03], self.evaluate(w), atol=0.01, rtol=0.01)


@test_util.run_all_in_graph_and_eager_modes
class DatasetIteratorTest(test_utils.TestCase):

  def testIteration(self):
    data = np.arange(100)
    ds = tf.data.Dataset.from_tensor_slices(data)
    itr = eager_utils.dataset_iterator(ds)
    for d in data:
      self.assertEqual(np.array([d]),
                       self.evaluate(eager_utils.get_next(itr)))


if __name__ == '__main__':
  tf.test.main()
