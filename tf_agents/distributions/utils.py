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

"""Utilities related to distributions."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import inspect
from typing import Any, Mapping, Type, Text

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import tanh_bijector_stable
from tf_agents.specs import tensor_spec
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


def scale_distribution_to_spec(distribution, spec):
  """Scales the given distribution to the bounds of the given spec."""
  return SquashToSpecNormal(distribution, spec)


class SquashToSpecNormal(tfp.distributions.Distribution):
  """Scales an input normalized action distribution to match spec bounds.

  Unlike the normal distribution computed when NormalProjectionNetwork
  is called with scale_distribution=False, which merely squashes the mean
  of the distribution to within the action spec, this distribution scales the
  output distribution to ensure that the output action fits within the spec.

  This distribution also maintains the input normal distribution, and uses this
  distribution to compute the KL-divergence between two SquashToSpecNormal
  distributions provided that they were scaled by the same action spec.
  This is possible as KL divergence is invariant when both distributions are
  transformed using the same invertible function.

  Formally, let a be the action magnitude and b be the action mean. The
  squashing operation performs the following change of variables to the
  input distribution X:

  Y = a * tanh(X) + b

  Note that this is a change of variables as the function is invertible, with:

  X = tan((Y - b) / a), where Y in (b - a, b + a)
  """

  def __init__(self,
               distribution,
               spec,
               validate_args=False,
               name="SquashToSpecNormal"):
    """Constructs a SquashToSpecNormal distribution.

    Args:
      distribution: input normal distribution with normalized mean and std dev
      spec: bounded action spec from which to compute action ranges
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.
    """

    if not isinstance(
        distribution,
        (tfp.distributions.Normal, tfp.distributions.MultivariateNormalDiag)):
      raise ValueError("Input distribution must be a normal distribution, "
                       "got {} instead".format(distribution))
    self.action_means, self.action_magnitudes = common.spec_means_and_magnitudes(
        spec)
    # Parameters here describe the actor network's output, which is a normalized
    # distribution prior to squashing to the action spec.
    # This is necessary (and sufficient) in order for policy info to compare an
    # old policy to a new policy.
    parameters = {"loc": distribution.loc, "scale": distribution.scale}
    # The raw action distribution
    self.input_distribution = distribution

    bijectors = [
        tfp.bijectors.Shift(self.action_means)(
            tfp.bijectors.Scale(self.action_magnitudes)),
        tanh_bijector_stable.Tanh()
    ]
    bijector_chain = tfp.bijectors.Chain(bijectors)
    self._squashed_distribution = tfp.distributions.TransformedDistribution(
        distribution=distribution, bijector=bijector_chain)
    super(SquashToSpecNormal, self).__init__(
        dtype=distribution.dtype,
        reparameterization_type=distribution.reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=distribution.allow_nan_stats,
        parameters=parameters,
        # We let TransformedDistribution access _graph_parents since this class
        # is more like a baseclass than derived.
        graph_parents=(
            distribution._graph_parents +  # pylint: disable=protected-access
            bijector_chain.graph_parents),
        name=name)

  def kl_divergence(self, other, name="kl_divergence"):
    """Computes the KL Divergence between two SquashToSpecNormal distributions."""
    if not isinstance(other, SquashToSpecNormal):
      raise ValueError("other distribution should be of type "
                       "SquashToSpecNormal, got {}".format(other))
    if (np.any(self.action_means != other.action_means) or
        np.any(self.action_magnitudes != other.action_magnitudes)):
      raise ValueError("Other distribution does not have same action mean "
                       "and magnitude. This mean {}, this magnitude {}, "
                       "other mean {}, other magnitude {}.".format(
                           self.action_means, self.action_magnitudes,
                           other.action_means, other.action_magnitudes))
    return self.input_distribution.kl_divergence(other.input_distribution, name)

  def sample(self, sample_shape=(), seed=None, name="sample"):
    """Generates samples from the wrapped TransformedDistribution."""
    return self._squashed_distribution.sample(sample_shape, seed, name)

  def log_prob(self, value, name="log_prob"):
    """Computes log probability from the wrapped TransformedDistribution."""
    return self._squashed_distribution.log_prob(value, name)

  def prob(self, value, name="prob"):
    """Computes probability from the wrapped TransformedDistribution."""
    return self._squashed_distribution.prob(value, name)

  def stddev(self, name="stddev"):
    """Compute stddev of the SquashToSpecNormal distribution."""
    stddev = self.action_magnitudes * tf.tanh(self.input_distribution.stddev())
    return stddev

  def mode(self, name="mode"):
    """Compute mean of the SquashToSpecNormal distribution."""
    mean = self.action_magnitudes * tf.tanh(self.input_distribution.mode()) + \
        self.action_means
    return mean

  def mean(self, name="mean", **kwargs):
    """Compute mean of the SquashToSpecNormal distribution."""
    return self.mode(name)

  def event_shape_tensor(self, name="event_shape_tensor"):
    """Compute event shape tensor of the SquashToSpecNormal distribution."""
    return self._squashed_distribution.event_shape_tensor(name)

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    """Compute event shape tensor of the SquashToSpecNormal distribution."""
    return self._squashed_distribution.batch_shape_tensor(name)


class Params(object):
  """The (recursive) parameters of objects exposing the `parameters` property.

  This includes TFP `Distribution`, `Bijector`, and TF `LinearOperator`.

  `Params` objects are created with
  `tf_agents.distributions.utils.get_parameters`;
  `Params` can be converted back to original objects via
  `tf_agents.distributions.utils.make_from_parameters`.

  In-place edits of fields are allowed, and will not modify the original
  objects (with the exception of, e.g., reference objects like `tf.Variable`
  being modified in-place).

  The components of a `Params` object are: `type_` and `params`.

  - `type_` is the type of object.
  - `params` is a `dict` of the (non-default) non-tensor arguments passed to the
    object's `__init__`; and includes nests of Python objects, as well as other
    `Params` values representing "Param-representable" objects passed to init.

  A non-trivial example:

  ```python
  scale_matrix = tf.Variable([[1.0, 2.0], [-1.0, 0.0]])
  d = tfp.distributions.MultivariateNormalDiag(
      loc=[1.0, 1.0], scale_diag=[2.0, 3.0], validate_args=True)
  b = tfp.bijectors.ScaleMatvecLinearOperator(
      scale=tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix),
      adjoint=True)
  b_d = b(d)
  p = utils.get_parameters(b_d)
  ```

  Then `p` is:

  ```python
  Params(
      tfp.distributions.TransformedDistribution,
      params={
          "bijector": Params(
              tfp.bijectors.ScaleMatvecLinearOperator,
              params={"adjoint": True,
                      "scale": Params(
                          tf.linalg.LinearOperatorFullMatrix,
                          params={"matrix": scale_matrix})}),
          "distribution": Params(
              tfp.distributions.MultivariateNormalDiag,
              params={"validate_args": True,
                      "scale_diag": [2.0, 3.0],
                      "loc": [1.0, 1.0]})})
  ```

  This structure can be manipulated and/or converted back to a `Distribution`
  instance via `make_from_parameters`:

  ```python
  p.params["distribution"].params["loc"] = [0.0, 0.0]

  # The distribution `new_b_d` will be a MVN centered on `(0, 0)` passed through
  # the `ScaleMatvecLinearOperator` bijector.
  new_b_d = utils.make_from_parameters(p)
  ```
  """
  type_: Type[Any]  # Any class that has a .parameters.
  params: Mapping[Text, Any]

  def __str__(self):
    return "<Params: type={}, params={}>".format(self.type_, self.params)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return (isinstance(self, type(other))
            and self.type_ == other.type_
            and self.params == other.params)

  def __init__(self, type_, params):
    self.type_ = type_
    self.params = params


def get_parameters(value: Any) -> Params:
  """Creates a recursive `Params` object from `value`.

  The `Params` object can be converted back to an object of type `type(value)`
  with `make_from_parameters`.  For more details, see the docstring of
  `Params`.

  Args:
    value: Typically a user provides `tfp.Distribution`, `tfp.Bijector`, or
      `tf.linalg.LinearOperator`, but this can be any Python object.

  Returns:
    An instance of `Params`.

  Raises:
    TypeError: If `value.parameters` exists, is not `None`, but but is also not
       a `Mapping` (e.g. a `dict`).
  """
  parameters = getattr(value, "parameters", None)
  if not isinstance(parameters, Mapping):
    raise TypeError(
        "value.parameters is not available or is not a dict; "
        "value: {}; parameters: {}".format(value, parameters))
  type_ = type(value)
  params = {}

  def process_parameter(p):
    if getattr(p, "parameters", None) is not None:
      return get_parameters(p)
    else:
      return p

  if getattr(value, "parameters"):
    default_values = inspect.signature(type_).parameters.items()
    default_values = {
        k: v.default
        for (k, v) in default_values
        if v.default is not inspect.Parameter.empty
    }
    params = {
        k: tf.nest.map_structure(process_parameter, v)
        for k, v in value.parameters.items()
        if v is not default_values.get(k, None)
    }
    return Params(type(value), params)

  return value


def make_from_parameters(value: Params) -> Any:
  """Creates an instance of type `value.type_` with the parameters in `value`.

  For more details, see the docstrings for `get_parameters` and `Params`.

  This function may raise strange errors if `value` is a `Params` created from
  a badly constructed object (one which does not set `self._parameters`
  properly).  For example:

  ```python
  class MyBadlyConstructedDistribution(tfp.distributions.Categorical):
    def __init__(self, extra_arg, **kwargs):
      super().__init__(**kwargs)
      self._extra_arg = extra_arg

    ...
  ```

  To fix this, make sure `self._parameters` are properly set:

  ```python
  class MyProperlyConstructedDistribution(tfp.distributions.Categorical):
    def __init__(self, extra_arg, **kwargs):
      super().__init__(**kwargs)
      # Ensure all arguments to `__init__` are in `self._parameters`.
      self._parameters = dict(extra_arg=extra_arg, **kwargs)
      self._extra_arg = extra_arg

    ...
  ```

  Args:
    value: A `Params` object; the output of `get_parameters` (or a
      modified version thereof).

  Returns:
    An instance of `value.type_`.

  Raises:
    Exception: If `value` is a `Params` object and the initializer of
      `value.type_` does not recognize accept the args structure given in
      `value.params`.  This can happen if, e.g., `value.type_.__init__` does not
      properly set `self._parameters` or `self.parameters` to match the
      arguments it expects.
  """
  def make_from_params_or_identity(v_):
    return make_from_parameters(v_) if isinstance(v_, Params) else v_

  params = {
      k: tf.nest.map_structure(make_from_params_or_identity, v)
      for k, v in value.params.items()
  }
  return value.type_(**params)


def parameters_to_dict(
    value: Params,
    tensors_only: bool = False) -> Mapping[Text, Any]:
  """Converts `value` to a nested `dict` (excluding all `type_` info).

  Sub-dicts represent `Params` objects; keys represent flattened nest structures
  in `value.params`.

  Example:

  ```python
  scale_matrix = tf.Variable([[1.0, 2.0], [-1.0, 0.0]])
  d = tfp.distributions.MultivariateNormalDiag(
      loc=[1.0, 1.0], scale_diag=[2.0, 3.0], validate_args=True)
  b = tfp.bijectors.ScaleMatvecLinearOperator(
      scale=tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix),
      adjoint=True)
  b_d = b(d)
  p = utils.get_parameters(b_d)
  params_dict = utils.parameters_to_dict(p, tensors_only)
  ```

  results in the nested dictionary, if `tensors_only=False`:

  ```python
  {
    "bijector": {"adjoint": True,
                 "scale": {"matrix": scale_matrix}},
    "distribution": {"validate_args": True,
                     # These are deeply nested because we passed lists
                     # intead of numpy arrays for `loc` and `scale_diag`.
                     "scale_diag:0": 2.0,
                     "scale_diag:1": 3.0,
                     "loc:0": 1.0,
                     "loc:1": 1.0}
  }
  ```

  results if `tensors_only=True`:

  ```python
  {
    "bijector": {"scale": {"matrix": scale_matrix}},
    "distribution": {},
  }
  ```

  The dictionary may then be modified or updated (e.g., in place), and converted
  back to a `Params` object using `merge_to_parameters_from_dict`.

  Args:
    value: The (possibly recursively defined) `Params`.
    tensors_only: Whether to include all parameters or only
      tensors and TypeSpecs.  If `True`, then only
      `tf.Tensor` and `tf.TypeSpec` leaf parameters are emitted.

  Returns:
    A `dict` mapping `value.params` to flattened key/value pairs.  Any
    sub-`Params` objects become nested dicts.
  """
  output_entries = {}

  # All tensors and TensorSpecs will be included for purposes of this test.
  is_tensor_or_spec = (
      lambda p: tf.is_tensor(p) or isinstance(p, tf.TypeSpec))

  def convert(key, p):
    if isinstance(p, Params):
      output_entries[key] = parameters_to_dict(p, tensors_only)
    elif not tensors_only or is_tensor_or_spec(p):
      output_entries[key] = p

  for k, v in value.params.items():
    if tf.nest.is_nested(v):
      flattened_params = nest_utils.flatten_with_joined_paths(v)
      for (param_k, param_v) in flattened_params:
        key = "{}:{}".format(k, param_k)
        convert(key, param_v)
    else:
      convert(k, v)

  return output_entries


def merge_to_parameters_from_dict(
    value: Params, params_dict: Mapping[Text, Any]) -> Params:
  """Merges dict matching data of `parameters_to_dict(value)` to a new `Params`.

  For more details, see the example below and the documentation of
  `parameters_to_dict`.

  Example:

  ```python
  scale_matrix = tf.Variable([[1.0, 2.0], [-1.0, 0.0]])
  d = tfp.distributions.MultivariateNormalDiag(
      loc=[1.0, 1.0], scale_diag=[2.0, 3.0], validate_args=True)
  b = tfp.bijectors.ScaleMatvecLinearOperator(
      scale=tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix),
      adjoint=True)
  b_d = b(d)
  p = utils.get_parameters(b_d)

  params_dict = utils.parameters_to_dict(p)
  params_dict["bijector"]["scale"]["matrix"] = new_scale_matrix

  new_params = utils.merge_to_parameters_from_dict(
    p, params_dict)

  # new_d is a `ScaleMatvecLinearOperator()(MultivariateNormalDiag)` with
  # a new scale matrix.
  new_d = utils.make_from_parameters(new_params)
  ```

  Args:
    value: A `Params` from which `params_dict` was derived.
    params_dict: A nested `dict` created by e.g. calling
      `parameters_to_dict(value)` and  modifying it to modify parameters.
      **NOTE** If any keys in the dict are missing, the "default" value in
      `value` is used instead.

  Returns:
    A new `Params` object which can then be turned into e.g. a
    `tfp.Distribution` via `make_from_parameters`.

  Raises:
    ValueError: If `params_dict` has keys missing from `value.params`.
    KeyError: If a subdict entry is missing for a nested value in
      `value.params`.
  """
  new_params = {}
  if params_dict is None:
    params_dict = {}

  processed_params = set()
  for k, v in value.params.items():
    # pylint: disable=cell-var-from-loop
    visited = set()
    converted = set()

    def convert(params_k, p):
      if params_k is not None:
        params_key = "{}:{}".format(k, params_k)
        visited.add(params_key)
        params_dict_value = params_dict.get(params_key, None)
        if params_dict_value is not None:
          converted.add(params_key)
      else:
        params_key = k
        params_dict_value = params_dict.get(k, None)
      processed_params.add(params_key)
      if isinstance(p, Params):
        return merge_to_parameters_from_dict(p, params_dict_value)
      else:
        return params_dict_value if params_dict_value is not None else p
    # pylint: enable=cell-var-from-loop

    if tf.nest.is_nested(v):
      new_params[k] = nest_utils.map_structure_with_paths(convert, v)
      if converted and visited != converted:
        raise KeyError(
            "Only saw partial information from the dictionary for nested "
            "key '{}' in params_dict.  Entries provided: {}.  "
            "Entries required: {}"
            .format(k, sorted(converted), sorted(visited)))
    else:
      new_params[k] = convert(None, v)

  unvisited_params_keys = set(params_dict) - processed_params
  if unvisited_params_keys:
    raise ValueError(
        "params_dict had keys that were not part of value.params.  "
        "params_dict keys: {}, value.params processed keys: {}".format(
            sorted(params_dict.keys()), sorted(processed_params)))

  return Params(type_=value.type_, params=new_params)


def _check_no_tensors(parameters: Params):
  flat_params = tf.nest.flatten(parameters.params)
  for p in flat_params:
    if isinstance(p, Params):
      _check_no_tensors(p)
    if tf.is_tensor(p):
      raise TypeError(
          "Saw a `Tensor` value in parameters:\n  {}".format(parameters))


class DistributionSpecV2(object):
  """Describes a tfp.distribution.Distribution using nested parameters."""

  def __init__(self,
               event_shape: tf.TensorShape,
               dtype: tf.DType,
               parameters: Params):
    """Construct a `DistributionSpecV2` from a Distribution's properties.

    Note that the `parameters` used to create the spec should contain
    `tf.TypeSpec` objects instead of tensors.  We check for this.

    Args:
      event_shape: The distribution's `event_shape`.  This is the shape that
        `distribution.sample()` returns.  `distribution.sample(sample_shape)`
        returns tensors of shape `sample_shape + event_shape`.
      dtype: The distribution's `dtype`.
      parameters: The recursive parameters of the distribution, with
        tensors having directly been converted to `tf.TypeSpec` objects.

    Raises:
      TypeError: If for any entry `x` in `parameters`: `tf.is_tensor(x)`.
    """
    _check_no_tensors(parameters)
    self._event_shape = event_shape
    self._dtype = dtype
    self._parameters = parameters
    self._event_spec = tf.TensorSpec(shape=event_shape, dtype=dtype)

  @property
  def event_shape(self) -> tf.TensorShape:
    return self._event_shape

  @property
  def dtype(self) -> tf.DType:
    return self._dtype

  @property
  def event_spec(self) -> tf.TensorSpec:
    return self._event_spec

  @property
  def parameters(self) -> Params:
    return self._parameters

  def __eq__(self, other):
    return (isinstance(self, type(other))
            and self._event_shape == other._event_shape
            and self._dtype == other._dtype
            and self._parameters == other._parameters)

  def __str__(self):
    return ("<DistributionSpecV2: event_shape={}, dtype={}, parameters={}>"
            .format(self.event_shape, self.dtype, self.parameters))

  def __repr__(self):
    return str(self)


def assert_specs_are_compatible(
    network_output_spec: types.NestedTensorSpec,
    spec: types.NestedTensorSpec,
    message_prefix: str):
  """Checks that the output of `network.create_variables` matches a spec.

  Args:
    network_output_spec: The output of `network.create_variables`.
    spec: The spec we are matching to.
    message_prefix: The message prefix for error messages, used when the specs
      don't match.

  Raises:
    ValueError: If the specs don't match.
  """
  def to_event(s):
    return (s.event_spec if isinstance(s, DistributionSpecV2)
            else tensor_spec.from_spec(s))

  event_spec = tf.nest.map_structure(to_event, network_output_spec)

  nest_utils.assert_same_structure(
      event_spec,
      spec,
      message=("{}:\n{}\nvs.\n{}".format(message_prefix, event_spec, spec)))

  def compare_output_to_spec(s1, s2):
    if not s1.is_compatible_with(s2):
      raise ValueError("{}:\n{}\nvs.\n{}".format(message_prefix, event_spec,
                                                 spec))

  tf.nest.map_structure(compare_output_to_spec, event_spec, spec)
