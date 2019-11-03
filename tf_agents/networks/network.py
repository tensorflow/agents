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
import six

import tensorflow as tf

from tensorflow.keras import layers  # pylint: disable=unused-import
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.keras.engine import network as keras_network  # TF internal
from tensorflow.python.training.tracking import base  # TF internal
from tensorflow.python.util import tf_decorator  # TF internal
from tensorflow.python.util import tf_inspect  # TF internal
# pylint:enable=g-direct-tensorflow-import


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

    def _capture_init(self, *args, **kwargs):
      """Captures init args and kwargs and stores them into `_saved_kwargs`."""
      if len(args) > len(arg_spec.args) + 1:
        # Error case: more inputs than args.  Call init so that the appropriate
        # error can be raised to the user.
        init(self, *args, **kwargs)
      for i, arg in enumerate(args):
        # Add +1 to skip `self` in arg_spec.args.
        kwargs[arg_spec.args[1 + i]] = arg
      init(self, **kwargs)
      # Avoid auto tracking which prevents keras from tracking layers that are
      # passed as kwargs to the Network.
      with base.no_automatic_dependency_tracking_scope(self):
        setattr(self, "_saved_kwargs", kwargs)

    attrs["__init__"] = tf_decorator.make_decorator(init, _capture_init)
    return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)


@six.add_metaclass(_NetworkMeta)
class Network(keras_network.Network):
  """Base extension to Keras network to simplify copy operations."""

  def __init__(self, input_tensor_spec, state_spec, name):
    """Creates an instance of `Network`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      state_spec: A nest of `tensor_spec.TensorSpec` representing the state
        needed by the network. Use () if none.
      name: A string representing the name of the network.
    """
    super(Network, self).__init__(name=name)
    self._input_tensor_spec = input_tensor_spec
    self._state_spec = state_spec

  @property
  def state_spec(self):
    return self._state_spec

  @property
  def input_tensor_spec(self):
    """Returns the spec of the input to the network of type InputSpec."""
    return self._input_tensor_spec

  def create_variables(self, **kwargs):
    if not self.built:
      random_input = tensor_spec.sample_spec_nest(
          self.input_tensor_spec, outer_dims=(0,))
      random_state = tensor_spec.sample_spec_nest(
          self.state_spec, outer_dims=(0,))
      step_type = tf.zeros([time_step.StepType.FIRST], dtype=tf.int32)
      self.__call__(
          random_input, step_type=step_type, network_state=random_state,
          **kwargs)

  @property
  def variables(self):
    if not self.built:
      raise ValueError(
          "Network has not been built, unable to access variables.  "
          "Please call `create_variables` or apply the network first.")
    return super(Network, self).variables

  @property
  def trainable_variables(self):
    if not self.built:
      raise ValueError(
          "Network has not been built, unable to access variables.  "
          "Please call `create_variables` or apply the network first.")
    return super(Network, self).trainable_variables

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

  def _check_trainable_weights_consistency(self):
    """Check trainable weights count consistency.

    This method makes up for the missing method (b/143631010) of the same name
    in `keras.Network`, which is needed when calling `Network.summary()`. This
    method is a no op. If a Network wants to check the consistency of trainable
    weights, see `keras.Model._check_trainable_weights_consistency` as a
    reference.
    """
    # TODO(b/143631010): If recognized and fixed, remove this entire method.
    return


class DistributionNetwork(Network):
  """Base class for networks which generate Distributions as their output."""

  def __init__(self, input_tensor_spec, state_spec, output_spec, name):
    super(DistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)
    self._output_spec = output_spec

  @property
  def output_spec(self):
    return self._output_spec
