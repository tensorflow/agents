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

"""Utilities for for interacting with datasets of encoded examples of TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.utils import eager_utils
from tf_agents.utils import example_encoding
from tf_agents.utils import nest_utils

from tensorflow.core.protobuf import struct_pb2  # pylint:disable=g-direct-tensorflow-import  # TF internal
from tensorflow.python.saved_model import nested_structure_coder  # pylint:disable=g-direct-tensorflow-import  # TF internal

# File extension used when saving data specs to file
_SPEC_FILE_EXTENSION = '.spec'


def encode_spec_to_file(output_path, tensor_data_spec):
  """Save a tensor data spec to a tfrecord file.

  Args:
    output_path: The path to the TFRecord file which will contain the spec.
    tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
      shape of the non-batched Tensors.
  """
  spec = tensor_spec.from_spec(tensor_data_spec)
  signature_encoder = nested_structure_coder.StructureCoder()
  spec_proto = signature_encoder.encode_structure(spec)
  with tf.io.TFRecordWriter(output_path) as writer:
    writer.write(spec_proto.SerializeToString())


def parse_encoded_spec_from_file(input_path):
  """Returns the tensor data spec stored at a path.

  Args:
    input_path: The path to the TFRecord file which contains the spec.

  Returns:
    Trajectory spec.
  Raises:
    IOError: File at input path does not exist.
  """
  if not os.path.exists(input_path):
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
  signature_encoder = nested_structure_coder.StructureCoder()
  spec = signature_encoder.decode_proto(signature_proto)
  return spec


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
  """

  def __init__(self, output_path, tensor_data_spec):
    """Creates observer object.

    Args:
      output_path: The path to the TFRecords file.
      tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
        shape of the non-batched Tensors.

    Raises:
      ValueError: if the tensors and specs have incompatible dimensions or
      shapes.
    """
    self._array_data_spec = tensor_spec.to_nest_array_spec(tensor_data_spec)
    self._encoder = example_encoding.get_example_serializer(
        self._array_data_spec)
    # Two output files: a tfrecord file and a file with the serialized spec
    self.output_path = output_path
    self._writer = tf.io.TFRecordWriter(self.output_path)
    logging.info('Writing dataset to TFRecord at %s', self.output_path)
    # Save the tensor spec used to write the dataset to file
    spec_output_path = self.output_path + _SPEC_FILE_EXTENSION
    encode_spec_to_file(spec_output_path, tensor_data_spec)

  def write(self, *tensor_data):
    """Encodes and writes (to file) a batch of tensor data.

    Args:
      *tensor_data: (unpacked) list/tuple of batched Tensors.
    """
    data = [x.numpy() for x in tensor_data]
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

  def __del__(self):
    self.close()

  def __call__(self, data):
    """Wraps write() function into a TensorFlow op for eager execution."""
    flat_data = tf.nest.flatten(data)
    tf.py_function(self.write, flat_data, [], name='encoder_observer')


def load_tfrecord_dataset(dataset_files, buffer_size=1000, as_experience=False):
  """Loads a TFRecord dataset from file, sequencing samples as Trajectories.

  Args:
    dataset_files: List of paths to one or more datasets
    buffer_size: (int) number of bytes in the read buffer. 0 means no buffering.
    as_experience: (bool) Returns dataset as a pair of Trajectories. Samples
      will be shaped as if they had been pulled from a replay buffer with
      `num_steps=2`. These samples can be fed directly to agent's `train`
      method.

  Returns:
    A dataset of type tf.data.Dataset. Samples follow the dataset's spec nested
    structure. Samples are generated with a leading batch dim of 1
    (or 2 if as_experience is enabled).
  Raises:
    IOError: One or more of the dataset files does not exist.
  """

  specs = []
  for dataset_file in dataset_files:
    spec_path = dataset_file + _SPEC_FILE_EXTENSION
    dataset_spec = parse_encoded_spec_from_file(spec_path)
    specs.append(dataset_spec)
    if not all([dataset_spec == spec for spec in specs]):
      raise IOError('One or more of the encoding specs do not match.')
  decoder = example_encoding.get_example_decoder(specs[0])
  logging.info('Loading TFRecord dataset...')
  dataset = tf.data.TFRecordDataset(
      dataset_files,
      buffer_size=buffer_size,
      num_parallel_reads=len(dataset_files))

  def decode_fn(proto):
    """Decodes a proto object."""
    return decoder(proto)

  def decode_and_batch_fn(proto):
    """Decodes a proto object, and batch output tensors."""
    sample = decoder(proto)
    return nest_utils.batch_nested_tensors(sample)

  if as_experience:
    return dataset.map(decode_fn).batch(2)
  else:
    return dataset.map(decode_and_batch_fn)
