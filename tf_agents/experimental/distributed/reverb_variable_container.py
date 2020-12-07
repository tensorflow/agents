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

# Lint as: python3
"""Distributed tf.Variable store client for Reverb backend.

This is just a client. The server needs to be consructed and held separately.
"""

from typing import Iterable, Text

import tensorflow as tf

from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import lazy_loader
from tf_agents.utils import nest_utils

# Lazy loading since not all users have the reverb package installed.
reverb = lazy_loader.LazyLoader('reverb', globals(), 'reverb')

# By default we assume that there is only one table with this name.
DEFAULT_TABLE = 'variables'

# Usually train step and policy variables are stored in variable container under
# the following name.
TRAIN_STEP_KEY = 'train_step'
POLICY_KEY = 'policy_variables'


class ReverbVariableContainer(object):
  """Distributed in memory tf.Variable store based on Reverb.

  This is just a client. The server needs to be consructed and held separately.

  **Note:** The container stores nests of variables in dedicated tables in a
  Reverb server. It is assumed that the server is running, the tables for
  variable container exist and are configured properly (i.e. it have signature
  defined and the `max_size=1`).
  """

  def __init__(self,
               server_address: Text,
               table_names: Iterable[Text] = (DEFAULT_TABLE,)):
    """Initializes the class.

    Args:
      server_address: The address of the Reverb server.
      table_names: Table names. By default, it is assumed that only a single
        table is used with the name `variables`. Each table assumed to exist in
        the server, has signature defined, and set the capacity to 1.

    Raises:
      KeyError: If a table is not defined in the server, but listed in tables.
      TypeError: If no signature is provided for a table.
      ValueError: If the `max_size` of the table corresponding to table_name(s)
        on the server is not equal to 1.
    """
    server_info = reverb.Client(server_address).server_info()
    self._dtypes = {}
    for table in table_names:
      table_info = server_info[table]
      if table_info.max_size != 1:
        raise ValueError(
            'The max_size of the table {} is {} which different from the '
            'expected capacity 1.'.format(table, table_info.max_size))
      if not table_info.signature:
        raise TypeError('Signature is not defined for table {}.'.format(table))
      self._dtypes[table] = tf.nest.map_structure(lambda spec: spec.dtype,
                                                  table_info.signature)
    self._tf_client = reverb.TFClient(server_address)

  def push(self,
           values: types.NestedTensor,
           table: Text = DEFAULT_TABLE) -> None:
    """Pushes values into a Reverb table.

    Args:
      values: Nested structure of tensors.
      table: The name of the table.

    Raises:
      KeyError: If the table name is not provided during construction time.
      tf.errors.InvalidArgumentError: If the nested structure of the variable
        does not match the signature of the table. This includes structural
        differences (excluding the type differences of sequences in nest), and
        type differences.
    """
    if table not in self._dtypes:
      raise KeyError('Could not find the table {}. Available tables: {}'.format(
          table, self._dtypes.keys()))

    # Sequence type check is turned off in Reverb client allowing sequence type
    # differences in the signature. This is required to be able work with
    # policies loaded from file which often change tuple to e.g. `ListWrapper`.
    self._tf_client.insert(
        data=tf.nest.flatten(values),
        tables=tf.constant([table]),
        priorities=tf.constant([1.0], dtype=tf.float64))

  def pull(self, table: Text = DEFAULT_TABLE) -> types.NestedTensor:
    """Pulls values from a Reverb table and returns them as nested tensors."""
    sample = self._tf_client.sample(table, data_dtypes=[self._dtypes[table]])
    # The data is received in the form of a sequence. In the case of variable
    # container the sequence length is always one.
    return sample.data[0]

  def update(self,
             variables: types.NestedVariable,
             table: Text = DEFAULT_TABLE) -> None:
    """Updates variables using values pulled from a Reverb table.

    Args:
      variables: Nested structure of variables.
      table: The name of the table.

    Raises:
      KeyError: If the table name is not provided during construction time.
      ValueError: If the nested structure of the variable does not match the
        signature of the table. This includes structural differences (excluding
        the type differences of sequences in nest), and type differences.
    """
    self._assign(variables, self.pull(table))

  # TODO(b/157554434): Move this to `nest_utils`.
  @common.function
  def _assign(self,
              variables: types.NestedVariable,
              values: types.NestedTensor,
              check_types: bool = False) -> None:
    """Assigns the nested values to variables."""
    nest_utils.assert_same_structure(variables, values, check_types=check_types)
    for variable, value in zip(
        tf.nest.flatten(variables), tf.nest.flatten(values)):
      variable.assign(value)
