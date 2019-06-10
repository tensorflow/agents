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

"""Base extension to Keras network to simplify copy operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import sys
import six
import tensorflow as tf

from tensorflow.keras import layers  # pylint: disable=unused-import
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from tensorflow.python.keras.engine import network as keras_network  # TF internal
from tensorflow.python.util import tf_decorator  # TF internal
from tensorflow.python.util import tf_inspect  # TF internal


class _NetworkMeta(abc.ABCMeta):
  """Meta class for Network object.

  We mainly use this class to capture all args to `__init__` of all `Network`
  instances, and store them in `instance._saved_kwargs`.  This in turn is
  used by the `instance.copy` method.
  """

  def __new__(mcs, classname, baseclasses, attrs):
    """Control the creation of subclasses of the Network class.

    Args:
      classname: The name of the subclass being created.
      baseclasses: A tuple of parent classes.
      attrs: A dict mapping new attributes to their values.

    Returns:
      The class object.

    Raises:
      RuntimeError: if the class __init__ has *args in its signature.
    """
    if baseclasses[0] == keras_network.Network:
      # This is just Network below.  Return early.
      return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)

    init = attrs.get("__init__", None)

    if not init:
      # This wrapper class does not define an __init__.  When someone creates
      # the object, the __init__ of its parent class will be called.  We will
      # call that __init__ instead separately since the parent class is also a
      # subclass of Network.  Here just create the class and return.
      return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)

    arg_spec = tf_inspect.getargspec(init)
    if arg_spec.varargs is not None:
      raise RuntimeError(
          "%s.__init__ function accepts *args.  This is not allowed." %
          classname)

    def capture_init(self, *args, **kwargs):
      if len(args) > len(arg_spec.args) + 1:
        # Error case: more inputs than args.  Call init so that the appropriate
        # error can be raised to the user.
        init(self, *args, **kwargs)
      for i, arg in enumerate(args):
        # Add +1 to skip `self` in arg_spec.args.
        kwargs[arg_spec.args[1 + i]] = arg
      init(self, **kwargs)
      setattr(self, "_saved_kwargs", kwargs)

    attrs["__init__"] = tf_decorator.make_decorator(init, capture_init)
    return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)


@six.add_metaclass(_NetworkMeta)
class Network(keras_network.Network):
  """Base extension to Keras network to simplify copy operations."""

  def __init__(self, input_tensor_spec, state_spec, name):
    super(Network, self).__init__(name=name)
    self._input_tensor_spec = input_tensor_spec
    self._state_spec = state_spec

  @property
  def state_spec(self):
    return self._state_spec

  def _build(self):
    if not self.built and self.input_tensor_spec is not None:
      random_input = tensor_spec.sample_spec_nest(
          self.input_tensor_spec, outer_dims=(1,))
      step_type = tf.expand_dims(time_step.StepType.FIRST, 0)
      self.__call__(random_input, step_type, None)

  @property
  def input_tensor_spec(self):
    """Returns the spec of the input to the network of type InputSpec."""
    return self._input_tensor_spec

  @property
  def variables(self):
    """Return the variables for all the network layers.

    If the network hasn't been built, builds it on random input (generated
    using self._input_tensor_spec) to build all the layers and their variables.

    Raises:
      ValueError:  If the network fails to build.
    """
    try:
      self._build()
    except ValueError as e:
      traceback = sys.exc_info()[2]
      six.reraise(
          ValueError, "Failed to call build on the network when accessing "
          "variables. Message: {!r}.".format(e), traceback)
    return self.weights

  def copy(self, **kwargs):
    """Create a shallow copy of this network.

    **NOTE** Network layer weights are *never* copied.  This method recreates
    the `Network` instance with the same arguments it was initialized with
    (excepting any new kwargs).

    Args:
      **kwargs: Args to override when recreating this network.  Commonly
        overridden args include 'name'.

    Returns:
      A shallow copy of this network.
    """
    return type(self)(**dict(self._saved_kwargs, **kwargs))

  def __call__(self, inputs, *args, **kwargs):
    tf.nest.assert_same_structure(inputs, self.input_tensor_spec)
    return super(Network, self).__call__(inputs, *args, **kwargs)


class DistributionNetwork(Network):
  """Base class for networks which generate Distributions as their output."""

  def __init__(self, input_tensor_spec, state_spec, output_spec, name):
    super(DistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)
    self._output_spec = output_spec

  @property
  def output_spec(self):
    return self._output_spec
