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

"""Utilities for easily encoding nests of numpy arrays into example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import common
from tf_agents.utils import nest_utils


def get_example_encoder(spec):
  """Get example encoder function for the given spec.

  Given a spec, returns an example encoder function. The example encoder
  function takes a nest of np.array feature values as input and returns a
  TF Example proto.

  Example:
    spec = {
        'lidar': array_spec.ArraySpec((900,), np.float32),
        'joint_positions': {
            'arm': array_spec.ArraySpec((7,), np.float32),
            'hand': array_spec.BoundedArraySpec((3, 3), np.int32, -1, 1)
        },
    }

    example_encoder = get_example_encoder(spec)
    serialized = example_encoder({
        'lidar': np.zeros((900,), np.float32),
        'joint_positions': {
            'arm': np.array([0.0, 1.57, 0.707, 0.2, 0.0, -1.57, 0.0],
                            np.float32),
            'hand': np.ones((3, 3), np.int32)
        },
    })

  The returned example encoder function requires that the feature nest passed
  has the shape and exact dtype specified in the spec. For example, it is
  an error to pass an array with np.float64 dtype where np.float32 is expected.

  Args:
    spec: list/tuple/nest of ArraySpecs describing a single example.

  Returns:
    Function

    ```python
    encoder(features_nest of np.arrays) -> tf.train.Example
    ```
  """
  # pylint: disable=g-complex-comprehension
  feature_encoders = [(path, _get_feature_encoder(spec.shape, spec.dtype))
                      for (path,
                           spec) in nest_utils.flatten_with_joined_paths(spec)]

  # pylint: enable=g-complex-comprehension

  def _example_encoder(features_nest):
    flat_features = tf.nest.flatten(features_nest)
    feature_dict = {
        path: feature_encoder(feature)
        for feature, (path,
                      feature_encoder) in zip(flat_features, feature_encoders)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

  return _example_encoder


def get_example_serializer(spec):
  """Returns string serializer of example protos."""
  encoder = get_example_encoder(spec)
  return lambda features_nest: encoder(features_nest).SerializeToString()


def get_example_decoder(example_spec, batched=False):
  """Get an example decoder function for a nested spec.

  Given a spec, returns an example decoder function. The decoder function parses
  string serialized example protos into tensors according to the given spec.

  Args:
    example_spec: list/tuple/nest of ArraySpecs describing a single example.
    batched: Boolean indicating if the decoder will receive batches of
      serialized data.

  Returns:
    Function

    ```python
    decoder(serialized_proto: tf.tensor[string]) -> example_spec nest of tensors
    ```
  """
  features_dict = {}
  parsers = []

  for (path, spec) in nest_utils.flatten_with_joined_paths(example_spec):
    feature, parser = _get_feature_parser(spec.shape, spec.dtype)
    features_dict[path] = feature
    parsers.append((path, parser))

  def _example_decoder(serialized):
    """Parses string serialized example protos into tensors."""
    if batched:
      raw_features = tf.io.parse_example(
          serialized=serialized, features=features_dict)

      decoded_features = []

      dtypes = [s.dtype for s in tf.nest.flatten(example_spec)]
      for (path, parser), dtype in zip(parsers, dtypes):
        decoded_features.append(
            tf.map_fn(parser, raw_features[path], dtype=dtype))

      return tf.nest.pack_sequence_as(example_spec, decoded_features)
    else:
      raw_features = tf.io.parse_single_example(
          serialized=serialized, features=features_dict)
      return tf.nest.pack_sequence_as(
          example_spec,
          [parser(raw_features[path]) for path, parser in parsers])

  return _example_decoder


def _validate_shape(shape):
  """Check that shape is a valid array shape."""
  if not isinstance(shape, collections.Iterable):
    raise TypeError('shape must be a tuple or other iterable object, not %s' %
                    type(shape).__name__)

  validated_shape = []
  for i, dim in enumerate(shape):
    if not dim or dim != int(dim) or int(dim) <= 0:
      raise ValueError(
          'Dimension %d is invalid in %s, expected positive int' % (i, shape))
    validated_shape.append(int(dim))

  return tuple(validated_shape)


def _validate_dtype(dtype):
  """Check that dtype is supported by tf.decode_raw."""
  dtype = tf.as_dtype(dtype)
  supported_dtypes = (tf.half, tf.float32, tf.float64, tf.uint8, tf.int8,
                      tf.uint16, tf.int16, tf.int32, tf.int64)
  if dtype not in supported_dtypes:
    raise ValueError('%s is not supported, dtype must be one of %s' %
                     (dtype.name, ', '.join(d.name for d in supported_dtypes)))
  return dtype


def _check_shape_and_dtype(value, shape, dtype):
  """Check that `value` has expected shape and dtype."""
  value_dtype = tf.as_dtype(value.dtype.newbyteorder('N'))
  if shape != value.shape or dtype != value_dtype:
    raise ValueError('Expected shape %s of %s, got: shape %s of %s' %
                     (shape, dtype.name, value.shape, value_dtype.name))


@gin.configurable
def _get_feature_encoder(shape, dtype, compress_image=False, image_quality=95):
  """Get feature encoder function for shape and dtype.

  Args:
    shape: An array shape
    dtype: A list of dtypes.
    compress_image: Whether to compress image. It is assumed that any uint8
      tensor of rank 3 with shape (w,h,3) is an image.
    image_quality: An optional int. Defaults to 95. Quality of the compression
      from 0 to 100 (higher is better and slower).

  Returns:
    A tf.train.Feature encoder.
  """
  shape = _validate_shape(shape)
  dtype = _validate_dtype(dtype)

  if compress_image and len(shape) == 3 and shape[2] == 3 and dtype == tf.uint8:
    if not common.has_eager_been_enabled():
      raise ValueError('Only supported in TF2.x.')
    def _encode_to_jpeg_bytes_list(value):
      value = tf.io.encode_jpeg(value, quality=image_quality)

      return tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[value.numpy()]))

    return _encode_to_jpeg_bytes_list

  if dtype == tf.float32:  # Serialize float32 to FloatList.

    def _encode_to_float_list(value):
      value = np.asarray(value)
      _check_shape_and_dtype(value, shape, dtype)
      return tf.train.Feature(
          float_list=tf.train.FloatList(
              value=value.flatten(order='C').tolist()))

    return _encode_to_float_list
  elif dtype == tf.int64:  # Serialize int64 to Int64List.

    def _encode_to_int64_list(value):
      value = np.asarray(value)
      _check_shape_and_dtype(value, shape, dtype)
      return tf.train.Feature(
          int64_list=tf.train.Int64List(
              value=value.flatten(order='C').tolist()))

    return _encode_to_int64_list
  else:  # Serialize anything else to BytesList in little endian order.
    le_dtype = dtype.as_numpy_dtype(0).newbyteorder('L')

    def _encode_to_bytes_list(value):
      value = np.asarray(value)
      _check_shape_and_dtype(value, shape, dtype)
      bytes_list_value = np.require(
          value, dtype=le_dtype, requirements='C').tostring()
      return tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[bytes_list_value]))

    return _encode_to_bytes_list


@gin.configurable
def _get_feature_parser(shape, dtype, compress_image=False):
  """Get tf.train.Features entry and decoder function for parsing feature.

  Args:
    shape: An array shape
    dtype: A list of dtypes.
    compress_image: Whether to decompress image. It is assumed that any uint8
      tensor of rank 3 with shape (w,h,3) is an image.
      If the tensor was compressed in the encoder, it needs to be decompressed.

  Returns:
    A tuple containing tf.io.FixedLenFeature decoder and decode function.
  """
  shape = _validate_shape(shape)
  dtype = _validate_dtype(dtype)

  if compress_image and len(shape) == 3 and shape[2] == 3 and dtype == tf.uint8:
    return (tf.io.FixedLenFeature(shape=[], dtype=tf.string), tf.io.decode_jpeg)

  if dtype == tf.float32:
    return (tf.io.FixedLenFeature(shape=shape, dtype=tf.float32), lambda x: x)
  elif dtype == tf.int64:
    return (tf.io.FixedLenFeature(shape=shape, dtype=tf.int64), lambda x: x)

  def decode(x):
    return tf.reshape(tf.io.decode_raw(x, dtype), shape)

  return (tf.io.FixedLenFeature(shape=[], dtype=tf.string), decode)
  # pylint: enable=g-long-lambda
