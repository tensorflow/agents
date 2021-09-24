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

"""Base extension to Keras network to simplify copy operations."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
import typing

import six

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import utils as distribution_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import keras as tf_keras  # TF internal
from tensorflow.python import util as tf_util  # TF internal
from tensorflow.python.training.tracking import base  # TF internal
# pylint: enable=g-direct-tensorflow-import


_ = tf.keras.layers  # Force loading of keras layers.
layer_utils = tf_keras.utils.layer_utils
tf_decorator = tf_util.tf_decorator
tf_inspect = tf_util.tf_inspect


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
    if baseclasses[0] == tf.keras.layers.Layer:
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
      # Convert to a canonical kwarg format.
      kwargs = tf_inspect.getcallargs(init, self, *args, **kwargs)
      kwargs.pop("self")
      init(self, **kwargs)
      # Avoid auto tracking which prevents keras from tracking layers that are
      # passed as kwargs to the Network.
      with base.no_automatic_dependency_tracking_scope(self):
        setattr(self, "_saved_kwargs", kwargs)

    attrs["__init__"] = tf_decorator.make_decorator(init, _capture_init)
    return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)


@six.add_metaclass(_NetworkMeta)
class Network(tf.keras.layers.Layer):
  """A class used to represent networks used by TF-Agents policies and agents.

  The main differences between a TF-Agents Network and a Keras Layer include:
  networks keep track of their underlying layers, explicitly represent RNN-like
  state in inputs and outputs, and simplify variable creation and clone
  operations.

  When calling a network `net`, typically one passes data through it via:

  ```python
  outputs, next_state = net(observation, network_state=...)
  outputs, next_state = net(observation, step_type=..., network_state=...)
  outputs, next_state = net(observation)  # net.call must fill an empty state
  outputs, next_state = net(observation, step_type=...)
  outputs, next_state = net(
      observation, step_type=..., network_state=..., learning=...)
  ```

  etc.

  To force construction of a network's variables:
  ```python
  net.create_variables()
  net.create_variables(input_tensor_spec=...)  # To provide an input spec
  net.create_variables(training=True)  # Provide extra kwargs
  net.create_variables(input_tensor_spec, training=True)
  ```

  To create a copy of the network:
  ```python
  cloned_net = net.copy()
  cloned_net.variables  # Raises ValueError: cloned net does not share weights.
  cloned_net.create_variables(...)
  cloned_net.variables  # Now new variables have been created.
  ```
  """

  # TODO(b/156314975): Rename input_tensor_spec to input_spec.
  def __init__(self, input_tensor_spec=None, state_spec=(), name=None):
    """Creates an instance of `Network`.

    Args:
      input_tensor_spec: A nest of `tf.TypeSpec` representing the
        input observations.  Optional.  If not provided, `create_variables()`
        will fail unless a spec is provided.
      state_spec: A nest of `tensor_spec.TensorSpec` representing the state
        needed by the network. Default is `()`, which means no state.
      name: (Optional.) A string representing the name of the network.
    """
    # Disable autocast because it may convert bfloats to other types, breaking
    # our spec checks.
    super(Network, self).__init__(name=name, autocast=False)
    common.check_tf1_allowed()

    # Required for summary() to work.
    self._is_graph_network = False

    self._input_tensor_spec = (
        tensor_spec.from_spec(input_tensor_spec)
        if input_tensor_spec is not None
        else None)
    # NOTE(ebrevdo): Would have preferred to call this output_tensor_spec, but
    # looks like keras.Layer already reserves that one.
    self._network_output_spec = None
    self._state_spec = tensor_spec.from_spec(state_spec)

  @property
  def state_spec(self):
    return self._state_spec

  @property
  def input_tensor_spec(self):
    """Returns the spec of the input to the network of type InputSpec."""
    return self._input_tensor_spec

  def create_variables(self, input_tensor_spec=None, **kwargs):
    """Force creation of the network's variables.

    Return output specs.

    Args:
      input_tensor_spec: (Optional).  Override or provide an input tensor spec
        when creating variables.
      **kwargs: Other arguments to `network.call()`, e.g. `training=True`.

    Returns:
      Output specs - a nested spec calculated from the outputs (excluding any
      batch dimensions).  If any of the output elements is a tfp `Distribution`,
      the associated spec entry returned is a `DistributionSpec`.

    Raises:
      ValueError: If no `input_tensor_spec` is provided, and the network did
        not provide one during construction.
    """
    if self._network_output_spec is not None:
      return self._network_output_spec
    if self._input_tensor_spec is None:
      self._input_tensor_spec = input_tensor_spec
    input_tensor_spec = self._input_tensor_spec
    if input_tensor_spec is None:
      raise ValueError(
          "Unable to create_variables: no input_tensor_spec provided, and "
          "Network did not define one.")

    random_input = tensor_spec.sample_spec_nest(
        input_tensor_spec, outer_dims=(1,))
    initial_state = self.get_initial_state(batch_size=1)
    step_type = tf.fill((1,), time_step.StepType.FIRST)
    outputs = self.__call__(
        random_input,
        step_type=step_type,
        network_state=initial_state,
        **kwargs)

    def _calc_unbatched_spec(x):
      """Build Network output spec by removing previously added batch dimension.

      Args:
        x: tfp.distributions.Distribution or Tensor.
      Returns:
        Specs without batch dimension representing x.
      """
      if isinstance(x, tfp.distributions.Distribution):
        parameters = distribution_utils.get_parameters(x)
        parameter_specs = _convert_to_spec_and_remove_singleton_batch_dim(
            parameters, outer_ndim=1)
        return distribution_utils.DistributionSpecV2(
            event_shape=x.event_shape, dtype=x.dtype,
            parameters=parameter_specs)
      else:
        return tensor_spec.remove_outer_dims_nest(
            tf.type_spec_from_value(x), num_outer_dims=1)

    self._network_output_spec = tf.nest.map_structure(
        _calc_unbatched_spec, outputs[0])
    return self._network_output_spec

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

  @property
  def layers(self):
    """Get the list of all (nested) sub-layers used in this Network."""
    return list(self._flatten_layers(include_self=False, recursive=False))

  def get_layer(self, name=None, index=None):
    """Retrieves a layer based on either its name (unique) or index.

    If `name` and `index` are both provided, `index` will take precedence.
    Indices are based on order of horizontal graph traversal (bottom-up).

    Args:
        name: String, name of layer.
        index: Integer, index of layer.

    Returns:
        A layer instance.

    Raises:
        ValueError: In case of invalid layer name or index.
    """
    if index is not None and name is not None:
      raise ValueError("Provide only a layer name or a layer index.")

    if index is not None:
      if len(self.layers) <= index:
        raise ValueError("Was asked to retrieve layer at index " + str(index) +
                         " but model only has " + str(len(self.layers)) +
                         " layers.")
      else:
        return self.layers[index]

    if name is not None:
      for layer in self.layers:
        if layer.name == name:
          return layer
      raise ValueError("No such layer: " + name + ".")

  def summary(self, line_length=None, positions=None, print_fn=None):
    """Prints a string summary of the network.

    Args:
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements
            in each line. If not provided,
            defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use. Defaults to `print`.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.

    Raises:
        ValueError: if `summary()` is called before the model is built.
    """
    if not self.built:
      raise ValueError("This model has not yet been built. "
                       "Build the model first by calling `build()` or "
                       "`__call__()` with some data, or `create_variables()`.")
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)

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
    """A wrapper around `Network.call`.

    A typical `call` method in a class subclassing `Network` will have a
    signature that accepts `inputs`, as well as other `*args` and `**kwargs`.
    `call` can optionally also accept `step_type` and `network_state`
    (if `state_spec != ()` is not trivial).  e.g.:

    ```python
    def call(self,
             inputs,
             step_type=None,
             network_state=(),
             training=False):
        ...
        return outputs, new_network_state
    ```

    We will validate the first argument (`inputs`)
    against `self.input_tensor_spec` if one is available.

    If a `network_state` kwarg is given it is also validated against
    `self.state_spec`.  Similarly, the return value of the `call` method is
    expected to be a tuple/list with 2 values:  `(output, new_state)`.
    We validate `new_state` against `self.state_spec`.

    If no `network_state` kwarg is given (or if empty `network_state = ()` is
    given, it is up to `call` to assume a proper "empty" state, and to
    emit an appropriate `output_state`.

    Args:
      inputs: The input to `self.call`, matching `self.input_tensor_spec`.
      *args: Additional arguments to `self.call`.
      **kwargs: Additional keyword arguments to `self.call`.
        These can include `network_state` and `step_type`.  `step_type` is
        required if the network's `call` requires it. `network_state` is
        required if the underlying network's `call` requires it.

    Returns:
      A tuple `(outputs, new_network_state)`.
    """
    if self.input_tensor_spec is not None:
      nest_utils.assert_matching_dtypes_and_inner_shapes(
          inputs,
          self.input_tensor_spec,
          allow_extra_fields=True,
          caller=self,
          tensors_name="`inputs`",
          specs_name="`input_tensor_spec`")

    call_argspec = tf_inspect.getargspec(self.call)

    # Convert *args, **kwargs to a canonical kwarg representation.
    normalized_kwargs = tf_inspect.getcallargs(
        self.call, inputs, *args, **kwargs)
    # TODO(b/156315434): Rename network_state to just state.
    network_state = normalized_kwargs.get("network_state", None)
    normalized_kwargs.pop("self", None)

    if common.safe_has_state(network_state):
      nest_utils.assert_matching_dtypes_and_inner_shapes(
          network_state,
          self.state_spec,
          allow_extra_fields=True,
          caller=self,
          tensors_name="`network_state`",
          specs_name="`state_spec`")

    if "step_type" not in call_argspec.args and not call_argspec.keywords:
      normalized_kwargs.pop("step_type", None)

    # network_state can be a (), None, Tensor or NestedTensors.
    if (not tf.is_tensor(network_state)
        and network_state in (None, ())
        and "network_state" not in call_argspec.args
        and not call_argspec.keywords):
      normalized_kwargs.pop("network_state", None)

    outputs, new_state = super(Network, self).__call__(**normalized_kwargs)  # pytype: disable=attribute-error  # typed-keras

    nest_utils.assert_matching_dtypes_and_inner_shapes(
        new_state,
        self.state_spec,
        allow_extra_fields=True,
        caller=self,
        tensors_name="`new_state`",
        specs_name="`state_spec`")

    return outputs, new_state

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

  def get_initial_state(self, batch_size=None):
    """Returns an initial state usable by the network.

    Args:
      batch_size: Tensor or constant: size of the batch dimension. Can be None
        in which case not dimensions gets added.

    Returns:
      A nested object of type `self.state_spec` containing properly
      initialized Tensors.
    """
    return self._get_initial_state(batch_size)

  def _get_initial_state(self, batch_size):
    """Returns the initial state of the policy network.

    Args:
      batch_size: A constant or Tensor holding the batch size. Can be None, in
        which case the state will not have a batch dimension added.

    Returns:
      A nest of zero tensors matching the spec of the policy network state.
    """
    return tensor_spec.zero_spec_nest(
        self._state_spec,
        outer_dims=None if batch_size is None else [batch_size])


class DistributionNetwork(Network):
  """Base class for networks which generate Distributions as their output."""

  def __init__(self, input_tensor_spec, state_spec, output_spec, name):
    super(DistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)
    self._output_spec = output_spec

  @property
  def output_spec(self):
    return self._output_spec


def _is_layer(obj):
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, tf.keras.layers.Layer).
  return hasattr(obj, "_is_layer") and not isinstance(obj, type)


def _filter_empty_layer_containers(layer_list):
  """Remove empty layer containers."""
  existing = object_identity.ObjectIdentitySet()
  to_visit = layer_list[::-1]
  while to_visit:
    obj = to_visit.pop()
    if obj in existing:
      continue
    existing.add(obj)
    if _is_layer(obj):
      yield obj
    else:
      sub_layers = getattr(obj, "layers", None) or []

      # Trackable data structures will not show up in ".layers" lists, but
      # the layers they contain will.
      to_visit.extend(sub_layers[::-1])


def _convert_to_spec_and_remove_singleton_batch_dim(
    parameters: distribution_utils.Params,
    outer_ndim: int) -> distribution_utils.Params:
  """Convert a `Params` object of tensors to one containing unbatched specs.

  Note: The `Params` provided to this function are typically contain tensors
  generated by Layers and therefore containing an outer singleton dimension.

  Since TF-Agents specs exclude batch and time prefixes, here we need to
  remove the singleton batch dimension from the specs created by these
  input tensors.

  Args:
    parameters: Distribution parameters, including input tensors.
    outer_ndim: Number of singleton outer dimensions expected in
      tensors found in `parameters`.

  Returns:
    A `Params` object contanining `tf.TypeSpec` in place of tensors, with
    up to `outer_ndim` outer singleton dimensions removed.
  """
  def _maybe_convert_to_spec(p):
    if isinstance(p, distribution_utils.Params):
      return _convert_to_spec_and_remove_singleton_batch_dim(p, outer_ndim)
    elif tf.is_tensor(p):
      return tensor_spec.remove_outer_dims_nest(
          tf.type_spec_from_value(p), num_outer_dims=outer_ndim)
    else:
      return p

  return distribution_utils.Params(
      type_=parameters.type_,
      params=tf.nest.map_structure(_maybe_convert_to_spec, parameters.params))


def create_variables(module: typing.Union[Network, tf.keras.layers.Layer],
                     input_spec: typing.Optional[types.NestedTensorSpec] = None,
                     **kwargs: typing.Any) -> types.NestedTensorSpec:
  """Create variables in `module` given `input_spec`; return `output_spec`.

  Here `module` can be a `tf_agents.networks.Network` or `Keras` layer.

  Args:
    module: The instance we would like to create layers on.
    input_spec: The input spec (excluding batch dimensions).
    **kwargs: Extra arguments to `module.__call__`, e.g. `training=True`.

  Returns:
    Output specs, a nested `tf.TypeSpec` describing the output signature.
    If `module` returns a `tfp.Distribution`, then the associated nested
    object is a `tf_agents.specs.DistributionSpecV2` (which is not a true
    `tf.TypeSpec` but contains enough information to create a nested
    `tf.TypeSpec` using `tf_agents.distributions.utils.parameters_to_dict`).

  Raises:
    ValueError: If `module` is a generic Keras layer but `input_spec is None`.
    TypeError: If `module` is a `tf.keras.layers.{RNN,LSTM,GRU,...}`.  These
      must be wrapped in `tf_agents.keras_layers.RNNWrapper`.
  """
  # NOTE(ebrevdo): As a side effect, for generic keras Layers (not Networks)
  # this method stores new hidden properties in `module`:
  # `_network_output_spec`, `_network_state_spec`,
  # - which internal TF-Agents libraries make use of.
  if isinstance(module, Network):
    return module.create_variables(input_spec, **kwargs)

  # Generic keras layer
  if input_spec is None:
    raise ValueError(
        "Module is a Keras layer; an input_spec is required but saw "
        "None: {}".format(module))

  if isinstance(module, tf.keras.layers.RNN):
    raise TypeError(
        "Keras RNN layers (non-cell layers) must be wrapped in "
        "tf_agents.keras_layers.RNNWrapper.  Layer: {}".format(module))

  maybe_spec = getattr(module, "_network_output_spec", None)
  if maybe_spec is not None:
    return maybe_spec

  # Has state outputs.
  recurrent_layer = getattr(module, "get_initial_state", None) is not None

  # Required input rank
  outer_ndim = _get_input_outer_ndim(module, input_spec)

  random_input = tensor_spec.sample_spec_nest(
      input_spec, outer_dims=(1,) * outer_ndim)

  if recurrent_layer:
    state = module.get_initial_state(random_input)

    def remove_singleton_batch_spec_dim(t):
      # Convert tensor to its type-spec, and remove the batch dimension
      # from the spec.
      spec = tf.type_spec_from_value(t)
      return nest_utils.remove_singleton_batch_spec_dim(spec, outer_ndim=1)
    state_spec = tf.nest.map_structure(remove_singleton_batch_spec_dim, state)

    outputs = module(random_input, state, **kwargs)
    # tf.keras.layers.{LSTMCell, ...} return (output, [state1, state2,...]).
    output = outputs[0]
  else:
    output = module(random_input, **kwargs)
    state_spec = ()

  def _calc_unbatched_spec(x):
    if isinstance(x, tfp.distributions.Distribution):
      parameters = distribution_utils.get_parameters(x)
      parameter_specs = _convert_to_spec_and_remove_singleton_batch_dim(
          parameters, outer_ndim=outer_ndim)
      return distribution_utils.DistributionSpecV2(
          event_shape=x.event_shape, dtype=x.dtype,
          parameters=parameter_specs)
    else:
      return tensor_spec.remove_outer_dims_nest(
          tf.type_spec_from_value(x), num_outer_dims=outer_ndim)

  # pylint: disable=protected-access
  module._network_output_spec = tf.nest.map_structure(_calc_unbatched_spec,
                                                      output)
  module._network_state_spec = state_spec

  return module._network_output_spec
  # pylint: enable=protected-access


def _get_input_outer_ndim(layer: tf.keras.layers.Layer,
                          input_spec: types.NestedTensorSpec) -> int:
  """Calculate or guess the number of batch (outer) ndims in `layer`."""
  if isinstance(layer, tf.keras.layers.RNN):
    raise TypeError(
        "Saw a tf.keras.layers.RNN layer nested inside e.g. a keras Sequential "
        "layer.  This is not directly supported.  Please wrap your layer "
        "inside a `tf_agents.keras_layers.RNNWrapper` or use "
        "`tf_agents.networks.Sequential`.  Layer: {}".format(layer))
  if isinstance(layer, tf.keras.layers.TimeDistributed):
    return 1 + _get_input_outer_ndim(layer.layer, input_spec)
  if isinstance(layer, tf.keras.Sequential):
    # We don't trust Sequential to give us the right thing if the first layer
    # is e.g. a TimeDistributed.
    return _get_input_outer_ndim(layer.layers[0], input_spec)

  layer_input_spec = layer.input_spec

  if layer_input_spec is None:
    return 1

  outer_ndim = layer_input_spec.ndim
  if outer_ndim is None:
    outer_ndim = layer_input_spec.min_ndim

  if outer_ndim is None:
    return 1

  if input_spec:
    input_spec = tf.nest.flatten(input_spec)[0]
    if outer_ndim >= input_spec.shape.ndims:
      # We can capture the "outer batch size" as the diff between the
      # expected input rank and the rank of the non-batched spec passed in the
      # input_spec.
      return outer_ndim - input_spec.shape.ndims

  # Empty input_spec.
  return 1


def get_state_spec(layer: tf.keras.layers.Layer) -> types.NestedTensorSpec:
  """Extracts the state spec from a layer.

  Args:
    layer: The layer to extract from; can be a `Network`.

  Returns:
    The state spec.

  Raises:
    TypeError: If `layer` is a subclass of `tf.keras.layers.RNN` (it must
      be wrapped by an `RNNWrapper` object).
    ValueError: If `layer` is a Keras layer and `create_variables` has
      not been called on it.
  """
  if isinstance(layer, Network):
    return layer.state_spec

  if isinstance(layer, tf.keras.layers.RNN):
    raise TypeError("RNN Layer must be wrapped inside "
                    "`tf_agents.keras_layers.RNNWrapper`: {}".format(layer))

  initial_state = getattr(layer, "get_initial_state", None)
  state_size = getattr(layer, "state_size", None)
  if initial_state is not None and state_size is None:
    raise ValueError(
        "Layer lacks a `state_size` property.  Unable to extract state "
        "spec: {}".format(layer))
  state_spec = ()
  if state_size is not None:
    state_spec = tf.nest.map_structure(
        lambda s: tf.TensorSpec(dtype=layer.dtype, shape=s), state_size)

  return state_spec
