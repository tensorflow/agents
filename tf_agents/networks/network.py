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

from tensorflow.keras import layers  # pylint: disable=unused-import
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
          "%s.__init__ function accepts *args.  This is not allowed."
          % classname)

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

  def __init__(self, observation_spec, action_spec, state_spec, name):
    super(Network, self).__init__(name=name)
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._state_spec = state_spec
    self._name = name

  @property
  def observation_spec(self):
    return self._observation_spec

  @property
  def action_spec(self):
    return self._action_spec

  @property
  def state_spec(self):
    return self._state_spec

  def copy(self, **kwargs):
    """Create a copy of this network.

    **NOTE** Network layer weights are *not* copied; just the parameters
    passed to the constructor of the original network.

    Args:
      **kwargs: Args to override when recreating this network.  Commonly
      overridden args include 'name'.

    Returns:
      A copy of this network.
    """
    return type(self)(**dict(self._saved_kwargs, **kwargs))
