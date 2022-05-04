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

"""Tests for variable container."""

from typing import Optional, Text, Tuple

from absl import logging
from absl.testing import parameterized
import numpy as np
import portpicker
import reverb
import tensorflow.compat.v2 as tf

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils


def _create_server(
    table: Text = reverb_variable_container.DEFAULT_TABLE,
    max_size: int = 1,
    signature: Optional[types.NestedTensorSpec] = (
        tf.TensorSpec((), tf.int64), {
            'var1': (tf.TensorSpec((2), tf.float64),),
            'var2': tf.TensorSpec((2, 1), tf.int32)
        })
) -> Tuple[reverb.Server, Text]:
  server = reverb.Server(
      tables=[
          reverb.Table(
              name=table,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=max_size,
              max_times_sampled=0,
              signature=signature)
      ],
      port=portpicker.pick_unused_port())
  return server, 'localhost:{}'.format(server.port)


def _create_nested_variable() -> types.NestedVariable:
  return (tf.Variable(0, dtype=tf.int64, shape=()), {
      'var1': (tf.Variable([1, 1], dtype=tf.float64, shape=(2,)),),
      'var2': tf.Variable([[2], [3]], dtype=tf.int32, shape=(2, 1))
  })


class ReverbVariableContainerTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self) -> None:
    super(ReverbVariableContainerTest, self).setUp()
    self._reverb_server, self._server_address = _create_server()

  def tearDown(self) -> None:
    if self._reverb_server:
      self._reverb_server.stop()
      self._reverb_server = None
    super(ReverbVariableContainerTest, self).tearDown()

  def test_init_raises_key_error_if_undefined_table_passed(self):
    server, server_address = _create_server(table='no_variables_table')
    with self.assertRaises(KeyError):
      reverb_variable_container.ReverbVariableContainer(server_address)
    server.stop()

  def test_init_raises_type_error_if_no_signature_of_a_table(self):
    server, server_address = _create_server(signature=None)
    with self.assertRaises(TypeError):
      reverb_variable_container.ReverbVariableContainer(server_address)
    server.stop()

  def test_init_raises_value_error_if_max_size_is_different_than_one(self):
    server, server_address = _create_server(max_size=2)
    with self.assertRaises(ValueError):
      reverb_variable_container.ReverbVariableContainer(server_address)
    server.stop()

  def test_push(self) -> None:
    # Prepare nested variables to push into the server.
    variables = _create_nested_variable()

    # Push the input to the server.
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    variable_container.push(variables)

    # Check the content of the server.
    self._assert_nested_variable_in_server()

  def test_push_with_not_exact_sequence_type_matching(self) -> None:
    # The second element (i.e the value of `var1`) was in a tuple in the
    # original signature, here we place it into a list.
    variables = (tf.Variable(0, dtype=tf.int64, shape=()), {
        'var1': [tf.Variable([1, 1], dtype=tf.float64, shape=(2,))],
        'var2': tf.Variable([[2], [3]], dtype=tf.int32, shape=(2, 1))
    })

    # Sequence type check is turned off by default allowing sequence type
    # differences in the signature. This is required to be able work with
    # policies loaded from file which often change tuple to e.g. `ListWrapper`.
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    variable_container.push(variables)

    # Check the content of the server.
    self._assert_nested_variable_in_server()

  def test_push_raises_key_error_on_unknown_table(self) -> None:
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    with self.assertRaises(KeyError):
      variable_container.push(tf.Variable(1), 'unknown_table')

  def test_push_raises_error_if_variable_struct_not_match(self) -> None:
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      variable_container.push(tf.Variable(1))

  def test_push_raises_error_if_variable_type_is_wrong(self) -> None:
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    # The first element has a type `tf.int64` in the signature, but here we
    # declare `tf.int32`.
    variables_with_wrong_type = (tf.Variable(-1, dtype=tf.int32, shape=()), {
        'var1': (tf.Variable([0, 0], dtype=tf.float64, shape=(2,)),),
        'var2': tf.Variable([[0], [0]], dtype=tf.int32, shape=(2, 1))
    })
    with self.assertRaises(tf.errors.InvalidArgumentError):
      variable_container.push(variables_with_wrong_type)

  def test_update(self) -> None:
    # Prepare some data in the Reverb server.
    self._push_nested_data()

    # Get the values from the server.
    variables = (tf.Variable(-1, dtype=tf.int64, shape=()), {
        'var1': (tf.Variable([0, 0], dtype=tf.float64, shape=(2,)),),
        'var2': tf.Variable([[0], [0]], dtype=tf.int32, shape=(2, 1))
    })

    # Update variables based on value pulled from the server.
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    variable_container.update(variables)

    # Check the values of the `variables`.
    self._assert_nested_variable_updated(variables)

  def test_update_with_not_exact_sequence_type_matching(self) -> None:
    # Prepare some data in the Reverb server.
    self._push_nested_data()

    # The second element (i.e the value of `var1`) was in a tuple in the
    # original signature, here we place it into a list.
    variables = (tf.Variable(-1, dtype=tf.int64, shape=()), {
        'var1': [tf.Variable([0, 0], dtype=tf.float64, shape=(2,))],
        'var2': tf.Variable([[0], [0]], dtype=tf.int32, shape=(2, 1))
    })

    # Sequence type check is turned off by default allowing sequence type
    # differences in the signature. This is required to be able work with
    # policies loaded from file which often change tuple to e.g. `ListWrapper`.
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    variable_container.update(variables)

    # Check the values of the `variables`.
    self._assert_nested_variable_updated(variables, check_nest_seq_types=False)

  def test_update_raises_key_error_on_unknown_table(self) -> None:
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    with self.assertRaises(KeyError):
      variable_container.update(tf.Variable(1), 'unknown_table')

  def test_update_raises_value_error_if_variable_struct_not_match(self) -> None:
    # Prepare some data in the Reverb server.
    self._push_nested_data()

    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    with self.assertRaises(ValueError):
      variable_container.update(tf.Variable(1))

  def test_update_raises_value_error_if_variable_type_is_wrong(self) -> None:
    # Prepare some data in the Reverb server.
    self._push_nested_data()

    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    # The first element has a type `tf.int64` in the signature, but here we
    # declare `tf.int32`.
    variables_with_wrong_type = (tf.Variable(-1, dtype=tf.int32, shape=()), {
        'var1': (tf.Variable([0, 0], dtype=tf.float64, shape=(2,)),),
        'var2': tf.Variable([[0], [0]], dtype=tf.int32, shape=(2, 1))
    })
    with self.assertRaises(ValueError):
      variable_container.update(variables_with_wrong_type)

  def test_pull_raises_key_error_on_unknown_table(self) -> None:
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    with self.assertRaises(KeyError):
      variable_container.pull('unknown_table')

  @parameterized.named_parameters(
      ('_default', tf.distribute.get_strategy()),
      ('_one_device', tf.distribute.OneDeviceStrategy('/cpu:0')),
      ('_mirrored', tf.distribute.MirroredStrategy(('/cpu:0', '/cpu:1'))))
  def test_push_under_distribute_strategy(
      self, strategy: tf.distribute.Strategy) -> None:
    # Prepare nested variables under strategy scope to push into the server.
    with strategy.scope():
      variables = _create_nested_variable()
    logging.info('Variables: %s', variables)

    # Push the input to the server.
    variable_container = reverb_variable_container.ReverbVariableContainer(
        self._server_address)
    variable_container.push(variables)

    # Check the content of the server.
    self._assert_nested_variable_in_server()

  def _push_nested_data(self, server_address: Optional[Text] = None) -> None:
    # Create Python client.
    address = server_address or self._server_address
    client = reverb.Client(address)
    with client.writer(1) as writer:
      writer.append([
          np.array(0, dtype=np.int64),
          np.array([1, 1], dtype=np.float64),
          np.array([[2], [3]], dtype=np.int32)
      ])
      writer.create_item(reverb_variable_container.DEFAULT_TABLE, 1, 1.0)
    self.assertEqual(
        client.server_info()[
            reverb_variable_container.DEFAULT_TABLE].current_size, 1)

  def _assert_nested_variable_in_server(self,
                                        server_address: Optional[Text] = None
                                       ) -> None:
    # Create Python client.
    address = server_address or self._server_address
    client = reverb.Client(address)
    self.assertEqual(
        client.server_info()[
            reverb_variable_container.DEFAULT_TABLE].current_size, 1)

    # Draw one sample from the server using the Python client.
    content = next(
        iter(client.sample(reverb_variable_container.DEFAULT_TABLE, 1)))[0].data  # pytype: disable=attribute-error  # strict_namedtuple_checks
    # Internally in Reverb the data is stored in the form of flat numpy lists.
    self.assertLen(content, 3)
    self.assertAllEqual(content[0], np.array(0, dtype=np.int64))
    self.assertAllClose(content[1], np.array([1, 1], dtype=np.float64))
    self.assertAllEqual(content[2], np.array([[2], [3]], dtype=np.int32))

  def _assert_nested_variable_updated(
      self,
      variables: types.NestedVariable,
      check_nest_seq_types: bool = True) -> None:
    # Prepare the exptected content of the variables.
    expected_values = (tf.constant(0, dtype=tf.int64, shape=()), {
        'var1': (tf.constant([1, 1], dtype=tf.float64, shape=(2,)),),
        'var2': tf.constant([[2], [3]], dtype=tf.int32, shape=(2, 1))
    })
    flat_expected_values = tf.nest.flatten(expected_values)

    # Assert that the variables have the same content as the expected values.
    # Meaning that the two nested structure have to be the same.
    self.assertIsNone(
        nest_utils.assert_same_structure(
            variables, expected_values, check_types=check_nest_seq_types))
    # And the values in `variables` have to be equal to (or close to, depending
    # on the component type) to the expected ones.
    flat_variables = tf.nest.flatten(variables)
    self.assertAllEqual(flat_variables[0], flat_expected_values[0])
    self.assertAllClose(flat_variables[1], flat_expected_values[1])
    self.assertAllEqual(flat_variables[2], flat_expected_values[2])


if __name__ == '__main__':
  multiprocessing.handle_test_main(test_utils.main)
