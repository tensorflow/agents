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

"""Utilities for for interacting with datasets of encoded examples of TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import eager_utils
from tf_agents.utils import example_encoding
from tf_agents.utils import nest_utils

from tensorflow.core.protobuf import struct_pb2  # pylint:disable=g-direct-tensorflow-import  # TF internal

# File extension used when saving data specs to file
_SPEC_FILE_EXTENSION = '.spec'


def encode_spec_to_file(output_path, tensor_data_spec):
  """Save a tensor data spec to a tfrecord file.

  Args:
    output_path: The path to the TFRecord file which will contain the spec.
    tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
      shape of the non-batched Tensors.
  """
  spec_proto = tensor_spec.to_proto(tensor_data_spec)
  with tf.io.TFRecordWriter(output_path) as writer:
    writer.write(spec_proto.SerializeToString())


def parse_encoded_spec_from_file(input_path):
  """Returns the tensor data spec stored at a path.

  Args:
    input_path: The path to the TFRecord file which contains the spec.

  Returns:
    `TensorSpec` nested structure parsed from the TFRecord file.
  Raises:
    IOError: File at input path does not exist.
  """
  if not tf.io.gfile.exists(input_path):
    raise IOError('Could not find spec file at %s.' % input_path)
  dataset = tf.data.TFRecordDataset(input_path, buffer_size=1)
  dataset_iterator = eager_utils.dataset_iterator(dataset)
  signature_proto_string = eager_utils.get_next(dataset_iterator)
  if tf.executing_eagerly():
    signature_proto = struct_pb2.StructuredValue.FromString(
        signature_proto_string.numpy())
  else:
    # In non-eager mode a session must be run in order to get the value
    with tf.Session() as sess:
      signature_proto_string_value = sess.run(signature_proto_string)
    signature_proto = struct_pb2.StructuredValue.FromString(
        signature_proto_string_value)
  return tensor_spec.from_proto(signature_proto)


class TFRecordObserver(object):
  """Observer for writing experience to TFRecord file.

  To use this observer, create an instance using a trajectory spec object
  and a dataset path:

  trajectory_spec = agent.collect_data_spec
  dataset_path = '/tmp/my_example_dataset'
  tfrecord_observer = TFRecordObserver(dataset_path, trajectory_spec)

  Then add it to the observers kwarg for the driver:

  collect_op = MyDriver(
    ...
    observers=[..., tfrecord_observer],
    num_steps=collect_steps_per_iteration).run()

  *Note*: Depending on your driver you may have to do
    `common.function(tfrecord_observer)` to handle the use of a callable with no
    return within a `tf.group` operation.
  """

  def __init__(self,
               output_path,
               tensor_data_spec,
               py_mode=False,
               compress_image=False,
               image_quality=95):
    """Creates observer object.

    Args:
      output_path: The path to the TFRecords file.
      tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
        shape of the non-batched Tensors.
      py_mode: Whether the observer is being used in a py_driver.
      compress_image: Whether to compress image. It is assumed that any uint8
        tensor of rank 3 with shape (w,h,c) is an image.
      image_quality: An optional int. Defaults to 95. Quality of the compression
        from 0 to 100 (higher is better and slower).

    Raises:
      ValueError: if the tensors and specs have incompatible dimensions or
      shapes.
    """
    self._py_mode = py_mode
    self._array_data_spec = tensor_spec.to_nest_array_spec(tensor_data_spec)
    self._encoder = example_encoding.get_example_serializer(
        self._array_data_spec,
        compress_image=compress_image,
        image_quality=image_quality)
    # Two output files: a tfrecord file and a file with the serialized spec
    self.output_path = output_path
    tf.io.gfile.makedirs(os.path.dirname(self.output_path))
    self._writer = tf.io.TFRecordWriter(self.output_path)
    logging.info('Writing dataset to TFRecord at %s', self.output_path)
    # Save the tensor spec used to write the dataset to file
    spec_output_path = self.output_path + _SPEC_FILE_EXTENSION
    encode_spec_to_file(spec_output_path, tensor_data_spec)

  def write(self, *data):
    """Encodes and writes (to file) a batch of data.

    Args:
      *data: (unpacked) list/tuple of batched np.arrays.
    """
    if self._py_mode:
      structured_data = data
    else:
      data = nest_utils.unbatch_nested_array(data)
      structured_data = tf.nest.pack_sequence_as(self._array_data_spec, data)
    self._writer.write(self._encoder(structured_data))

  def flush(self):
    """Manually flush TFRecord writer."""
    self._writer.flush()

  def close(self):
    """Close the TFRecord writer."""
    self._writer.close()
    logging.info('Closing TFRecord file at %s', self.output_path)

  def __call__(self, data):
    """If not in py_mode Wraps write() into a TF op for eager execution."""
    if self._py_mode:
      self.write(data)
    else:
      flat_data = tf.nest.flatten(data)
      tf.numpy_function(self.write, flat_data, [], name='encoder_observer')


def load_tfrecord_dataset(dataset_files,
                          buffer_size=1000,
                          as_experience=False,
                          as_trajectories=False,
                          add_batch_dim=True,
                          decoder=None,
                          num_parallel_reads=None,
                          compress_image=False,
                          spec=None):
  """Loads a TFRecord dataset from file, sequencing samples as Trajectories.

  Args:
    dataset_files: List of paths to one or more datasets
    buffer_size: (int) number of bytes in the read buffer. 0 means no buffering.
    as_experience: (bool) Returns dataset as a pair of Trajectories. Samples
      will be shaped as if they had been pulled from a replay buffer with
      `num_steps=2`. These samples can be fed directly to agent's `train`
      method.
    as_trajectories: (bool) Remaps the data into trajectory objects. This should
      be enabled when the resulting types must be trajectories as expected by
      agents.
    add_batch_dim: (bool) If True the data will have a batch dim of 1 to conform
      with the expected tensor batch convention. Set to false if you want to
      batch the data on your own.
    decoder: Optional, a custom decoder to use rather than using the default
      spec path.
    num_parallel_reads: Optional, number of parallel reads in the TFRecord
      dataset. If not specified, len(dataset_files) will be used.
    compress_image: Whether to decompress image. It is assumed that any uint8
      tensor of rank 3 with shape (w,h,c) is an image.
      If the tensor was compressed in the encoder, it needs to be decompressed.
    spec: Optional, the dataspec of the TFRecord dataset to be parsed. If not
      provided, parses the dataspec of the TFRecord directly.

  Returns:
    A dataset of type tf.data.Dataset. Samples follow the dataset's spec nested
    structure. Samples are generated with a leading batch dim of 1
    (or 2 if as_experience is enabled).
  Raises:
    IOError: One or more of the dataset files does not exist.
  """

  if not decoder:
    if spec is None:
      specs = []
      for dataset_file in dataset_files:
        spec_path = dataset_file + _SPEC_FILE_EXTENSION
        dataset_spec = parse_encoded_spec_from_file(spec_path)
        specs.append(dataset_spec)
        if not all([dataset_spec == spec for spec in specs]):
          raise IOError('One or more of the encoding specs do not match.')
      spec = specs[0]

    decoder = example_encoding.get_example_decoder(
        spec, compress_image=compress_image)

  logging.info('Loading TFRecord dataset...')
  if not num_parallel_reads:
    num_parallel_reads = len(dataset_files)
  dataset = tf.data.TFRecordDataset(
      dataset_files,
      buffer_size=buffer_size,
      num_parallel_reads=num_parallel_reads)

  def decode_fn(proto):
    """Decodes a proto object."""
    return decoder(proto)

  def decode_and_batch_fn(proto):
    """Decodes a proto object, and batch output tensors."""
    sample = decoder(proto)
    return nest_utils.batch_nested_tensors(sample)

  if as_experience:
    dataset = dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            2, drop_remainder=True)
  elif add_batch_dim:
    dataset = dataset.map(
        decode_and_batch_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    dataset = dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if as_trajectories:
    as_trajectories_fn = lambda sample: trajectory.Trajectory(*sample)
    dataset = dataset.map(
        as_trajectories_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset
