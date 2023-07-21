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

"""Network layer utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def _count_params(weights):
  """Count the total number of scalars composing the weights.

  Args:
      weights: An iterable containing the weights on which to compute params

  Returns:
      The total number of scalars composing the weights
  """
  unique_weights = {id(w): w for w in weights}.values()
  # Ignore TrackableWeightHandlers, which will not have a shape defined.
  unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
  weight_shapes = [w.shape.as_list() for w in unique_weights]
  standardized_weight_shapes = [
      [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
  ]
  return int(sum(np.prod(p) for p in standardized_weight_shapes))


def _weight_memory_size(weights):
  """Calculate the memory footprint for weights based on their dtypes.

  Args:
      weights: An iterable contains the weights to compute weight size.

  Returns:
      The total memory size (in Bytes) of the weights.
  """
  unique_weights = {id(w): w for w in weights}.values()

  total_memory_size = 0
  for w in unique_weights:
    # Ignore TrackableWeightHandlers, which will not have a shape defined.
    if not hasattr(w, "shape"):
      continue
    elif None in w.shape.as_list():
      continue
    weight_shape = np.prod(w.shape.as_list())
    per_param_size = w.dtype.size
    total_memory_size += weight_shape * per_param_size
  return total_memory_size


def _get_layer_index_bound_by_layer_name(model, layer_range=None):
  """Get the layer indexes from the model based on layer names.

  The layer indexes can be used to slice the model into sub models for
  display.

  Args:
      model: `tf.keras.Model` instance.
      layer_range: a list or tuple of 2 strings, the starting layer name and
        ending layer name (both inclusive) for the result. All layers will be
        included when `None` is provided.

  Returns:
      The index value of layer based on its unique name (layer_names).
      Output will be [first_layer_index, last_layer_index + 1].
  """
  if layer_range is not None:
    if len(layer_range) != 2:
      raise ValueError(
          "layer_range must be a list or tuple of length 2. Received: "
          f"layer_range = {layer_range} of length {len(layer_range)}"
      )
    if not isinstance(layer_range[0], str) or not isinstance(
        layer_range[1], str
    ):
      raise ValueError(
          "layer_range should contain string type only. "
          f"Received: {layer_range}"
      )
  else:
    return [0, len(model.layers)]

  lower_index = [
      idx
      for idx, layer in enumerate(model.layers)
      if re.match(layer_range[0], layer.name)
  ]
  upper_index = [
      idx
      for idx, layer in enumerate(model.layers)
      if re.match(layer_range[1], layer.name)
  ]

  if not lower_index or not upper_index:
    raise ValueError(
        "Passed layer_names do not match the layer names in the model. "
        f"Received: {layer_range}"
    )

  if min(lower_index) > max(upper_index):
    return [min(upper_index), max(lower_index) + 1]
  return [min(lower_index), max(upper_index) + 1]


def _readable_memory_size(weight_memory_size):
  """Convert the weight memory size (Bytes) to a readable string."""
  units = ["Byte", "KB", "MB", "GB", "TB", "PB"]
  scale = 1024
  for unit in units:
    if weight_memory_size / scale < 1:
      return "{:.2f} {}".format(weight_memory_size, unit)
    else:
      weight_memory_size /= scale
  return "{:.2f} {}".format(weight_memory_size, units[-1])


def _dtensor_variable_summary(weights):
  """Group and calculate DTensor based weights memory size.

  Since DTensor weights can be sharded across multiple device, the result
  will be grouped by the layout/sharding spec for the variables, so that
  the accurate per-device memory size can be calculated.

  Args:
      weights: An iterable contains the weights to compute weight size.

  Returns:
      total_weight_count, total_memory_size and per_sharing_spec_result which
      is a dict with normalized layout spec as key and tuple of weight count
      and weight size as value.
  """
  unique_weights = {id(w): w for w in weights}.values()
  total_weight_count = 0
  total_memory_size = 0
  per_sharing_spec_result = {}
  for w in unique_weights:
    # Ignore TrackableWeightHandlers, which will not have a shape defined.
    if not hasattr(w, "shape"):
      continue
    if not isinstance(w, tf.experimental.dtensor.DVariable):
      continue
    layout = w.layout
    # Remove all the duplication axis, and sort the column name.
    # 1D replicated and 2D replicated variable will still be fully
    # replicated, and [batch, model] sharding will have same memory
    # footprint as the [model, batch] layout.
    reduced_sharding_spec = list(sorted(set(layout.sharding_specs)))
    if tf.experimental.dtensor.UNSHARDED in reduced_sharding_spec:
      reduced_sharding_spec.remove(tf.experimental.dtensor.UNSHARDED)
    reduced_sharding_spec = tuple(reduced_sharding_spec)  # For dict key
    weight_count, memory_size = per_sharing_spec_result.get(
        reduced_sharding_spec, (0, 0)
    )
    reduced_weight_shape = np.prod(w.shape.as_list())
    per_param_size = w.dtype.size
    weight_count += reduced_weight_shape
    memory_size += reduced_weight_shape * per_param_size
    per_sharing_spec_result[reduced_sharding_spec] = (
        weight_count,
        memory_size,
    )
    total_weight_count += reduced_weight_shape
    total_memory_size += reduced_weight_shape * per_param_size
  return total_weight_count, total_memory_size, per_sharing_spec_result


def _print_dtensor_variable_summary(model, print_fn, line_length):
  """Prints dtensor variables summary."""
  if getattr(model, "_layout_map", None) is not None:  # pylint: disable=protected-access
    mesh = model._layout_map.get_default_mesh()  # pylint: disable=protected-access
  elif hasattr(model, "distribute_strategy") and hasattr(
      model.distribute_strategy, "_mesh"
  ):
    mesh = model.distribute_strategy._mesh  # pylint: disable=protected-access
  else:
    # Not running with DTensor
    mesh = None
  if mesh:
    (
        total_weight_count,
        total_memory_size,
        per_sharing_spec_result,
    ) = _dtensor_variable_summary(model.weights)
    total_per_device_memory_size = 0
    for sharding_spec in sorted(per_sharing_spec_result.keys()):  # pylint: disable=g-builtin-op
      count, memory_size = per_sharing_spec_result[sharding_spec]
      if len(sharding_spec) == 0:  # pylint: disable=g-explicit-length-test
        print_fn(
            f"{count} / {total_weight_count} params "
            f"({_readable_memory_size(memory_size)}) "
            "are fully replicated"
        )
        per_device_size = memory_size
      else:
        sharding_factor = np.prod([mesh.dim_size(s) for s in sharding_spec])
        per_device_size = memory_size / sharding_factor
        print_fn(
            f"{count} / {total_weight_count} params "
            f"({_readable_memory_size(memory_size)}) are sharded based "
            f"on spec '{sharding_spec}' and across {sharding_factor} "
            "devices."
        )
      total_per_device_memory_size += per_device_size
    print_fn(
        "Overall per device memory usage: "
        f"{_readable_memory_size(total_per_device_memory_size)}"
    )
    print_fn(
        "Overall sharding factor: {:.2f}".format(
            total_memory_size / total_per_device_memory_size
        )
    )
    print_fn("_" * line_length)


def print_summary(
    model,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
):
  """Prints a summary of a model.

  Args:
      model: Keras model instance.
      line_length: Total length of printed lines (e.g. set this to adapt the
        display to different terminal window sizes).
      positions: Relative or absolute positions of log elements in each line. If
        not provided, defaults to `[0.3, 0.6, 0.70, 1.]`.
      print_fn: Print function to use. It will be called on each line of the
        summary. You can set it to a custom function in order to capture the
        string summary. It defaults to `print` (prints to stdout).
      expand_nested: Whether to expand the nested models. If not provided,
        defaults to `False`.
      show_trainable: Whether to show if a layer is trainable. If not provided,
        defaults to `False`.
      layer_range: List or tuple containing two strings, the starting layer name
        and ending layer name (both inclusive), indicating the range of layers
        to be printed in the summary. The strings could also be regexes instead
        of an exact name. In this case, the starting layer will be the first
        layer that matches `layer_range[0]` and the ending layer will be the
        last element that matches `layer_range[1]`. By default (`None`) all
        layers in the model are included in the summary.
  """
  if print_fn is None:
    print_fn = print

  if model.__class__.__name__ == "Sequential":
    sequential_like = True
  elif not model._is_graph_network:  # pylint: disable=protected-access
    # We treat subclassed models as a simple sequence of layers, for logging
    # purposes.
    sequential_like = True
  else:
    sequential_like = True
    nodes_by_depth = model._nodes_by_depth.values()  # pylint: disable=protected-access
    nodes = []
    for v in nodes_by_depth:
      if (len(v) > 1) or (
          len(v) == 1 and len(tf.nest.flatten(v[0].keras_inputs)) > 1
      ):
        # if the model has multiple nodes
        # or if the nodes have multiple inbound_layers
        # the model is no longer sequential
        sequential_like = False
        break
      nodes += v
    if sequential_like:
      # search for shared layers
      for layer in model.layers:
        flag = False
        for node in layer._inbound_nodes:  # pylint: disable=protected-access
          if node in nodes:
            if flag:
              sequential_like = False
              break
            else:
              flag = True
        if not sequential_like:
          break

  relevant_nodes = None
  if sequential_like:
    line_length = line_length or 65
    positions = positions or [0.45, 0.85, 1.0]
    if positions[-1] <= 1:
      positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ["Layer (type)", "Output Shape", "Param #"]
  else:
    line_length = line_length or 98
    positions = positions or [0.3, 0.6, 0.70, 1.0]
    if positions[-1] <= 1:
      positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
    relevant_nodes = []
    for v in model._nodes_by_depth.values():  # pylint: disable=protected-access
      relevant_nodes += v

  if show_trainable:
    line_length += 11
    positions.append(line_length)
    to_display.append("Trainable")

  layer_range = _get_layer_index_bound_by_layer_name(model, layer_range)

  def print_row(fields, positions, nested_level=0):
    left_to_print = [str(x) for x in fields]
    while any(left_to_print):
      line = ""
      for col in range(len(left_to_print)):
        if col > 0:
          start_pos = positions[col - 1]
        else:
          start_pos = 0
        end_pos = positions[col]
        # Leave room for 2 spaces to delineate columns
        # we don't need any if we are printing the last column
        space = 2 if col != len(positions) - 1 else 0
        cutoff = end_pos - start_pos - space
        # Except for last col, offset by one to align the start of col
        if col != len(positions) - 1:
          cutoff -= 1
        if col == 0:
          cutoff -= nested_level
        fit_into_line = left_to_print[col][:cutoff]
        # For nicer formatting we line-break on seeing end of
        # tuple/dict etc.
        line_break_conditions = ("),", "},", "],", "',")
        candidate_cutoffs = [
            fit_into_line.find(x) + len(x)
            for x in line_break_conditions
            if fit_into_line.find(x) >= 0
        ]
        if candidate_cutoffs:
          cutoff = min(candidate_cutoffs)
          fit_into_line = fit_into_line[:cutoff]

        if col == 0:
          line += "|" * nested_level + " "
        line += fit_into_line
        line += " " * space if space else ""
        left_to_print[col] = left_to_print[col][cutoff:]

        # Pad out to the next position
        # Make space for nested_level for last column
        if nested_level and col == len(positions) - 1:
          line += " " * (positions[col] - len(line) - nested_level)
        else:
          line += " " * (positions[col] - len(line))
      line += "|" * nested_level
      print_fn(line)

  print_fn(f'Model: "{model.name}"')
  print_fn("_" * line_length)
  print_row(to_display, positions)
  print_fn("=" * line_length)

  def print_layer_summary(layer, nested_level=0):
    """Prints a summary for a single layer.

    Args:
        layer: target layer.
        nested_level: level of nesting of the layer inside its parent layer
          (e.g. 0 for a top-level layer, 1 for a nested layer).
    """
    try:
      output_shape = layer.output_shape
    except AttributeError:
      output_shape = "multiple"
    except RuntimeError:  # output_shape unknown in Eager mode.
      output_shape = "?"
    name = layer.name
    cls_name = layer.__class__.__name__
    if not layer.built and not getattr(layer, "_is_graph_network", False):
      # If a subclassed model has a layer that is not called in
      # Model.call, the layer will not be built and we cannot call
      # layer.count_params().
      params = "0 (unused)"
    else:
      params = layer.count_params()
    fields = [name + " (" + cls_name + ")", output_shape, params]

    if show_trainable:
      fields.append("Y" if layer.trainable else "N")

    print_row(fields, positions, nested_level)

  def print_layer_summary_with_connections(layer, nested_level=0):
    """Prints a summary for a single layer (including its connections).

    Args:
        layer: target layer.
        nested_level: level of nesting of the layer inside its parent layer
          (e.g. 0 for a top-level layer, 1 for a nested layer).
    """
    try:
      output_shape = layer.output_shape
    except AttributeError:
      output_shape = "multiple"
    connections = []
    for node in layer._inbound_nodes:  # pylint: disable=protected-access
      if relevant_nodes and node not in relevant_nodes:
        # node is not part of the current network
        continue

      for (
          inbound_layer,
          node_index,
          tensor_index,
          _,
      ) in node.iterate_inbound():
        connections.append(
            f"{inbound_layer.name}[{node_index}][{tensor_index}]"
        )

    name = layer.name
    cls_name = layer.__class__.__name__
    fields = [
        name + " (" + cls_name + ")",
        output_shape,
        layer.count_params(),
        connections,
    ]

    if show_trainable:
      fields.append("Y" if layer.trainable else "N")

    print_row(fields, positions, nested_level)

  def print_layer(layer, nested_level=0, is_nested_last=False):
    if sequential_like:
      print_layer_summary(layer, nested_level)
    else:
      print_layer_summary_with_connections(layer, nested_level)

    if expand_nested and hasattr(layer, "layers") and layer.layers:
      print_fn(
          "|" * (nested_level + 1)
          + "¯" * (line_length - 2 * nested_level - 2)
          + "|" * (nested_level + 1)
      )

      nested_layer = layer.layers
      is_nested_last = False
      for i in range(len(nested_layer)):
        if i == len(nested_layer) - 1:
          is_nested_last = True
        print_layer(nested_layer[i], nested_level + 1, is_nested_last)

      print_fn(
          "|" * nested_level
          + "¯" * (line_length - 2 * nested_level)
          + "|" * nested_level
      )

    if not is_nested_last:
      print_fn(
          "|" * nested_level
          + " " * (line_length - 2 * nested_level)
          + "|" * nested_level
      )

  for layer in model.layers[layer_range[0] : layer_range[1]]:
    print_layer(layer)
  print_fn("=" * line_length)

  if hasattr(model, "_collected_trainable_weights"):
    trainable_count = _count_params(model._collected_trainable_weights)  # pylint: disable=protected-access
    trainable_memory_size = _weight_memory_size(
        model._collected_trainable_weights  # pylint: disable=protected-access
    )
  else:
    trainable_count = _count_params(model.trainable_weights)
    trainable_memory_size = _weight_memory_size(model.trainable_weights)

  non_trainable_count = _count_params(model.non_trainable_weights)
  non_trainable_memory_size = _weight_memory_size(model.non_trainable_weights)  # pylint: disable=protected-access

  total_memory_size = trainable_memory_size + non_trainable_memory_size

  print_fn(
      f"Total params: {trainable_count + non_trainable_count} "
      f"({_readable_memory_size(total_memory_size)})"
  )
  print_fn(
      f"Trainable params: {trainable_count} "
      f"({_readable_memory_size(trainable_memory_size)})"
  )
  print_fn(
      f"Non-trainable params: {non_trainable_count} "
      f"({_readable_memory_size(non_trainable_memory_size)})"
  )
  print_fn("_" * line_length)

  _print_dtensor_variable_summary(model, print_fn, line_length)
