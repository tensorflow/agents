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
"""Utils for using replay buffers."""

import tensorflow as tf

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils import lazy_loader

# Lazy loading since not all users have the reverb package installed.
reverb = lazy_loader.LazyLoader('reverb', globals(), 'reverb')


# Default table creation function that creates a uniform sampling table
# and default paramenters.
def _create_uniform_table(table_name, data_spec, table_capacity=1000,
                          min_size_limiter_size=1):
  """Creates a uniform table with default parameters.

  Args:
    table_name: string name of the uniform sampling table
    data_spec: Spec for the data the table will hold.
    table_capacity:  capacity of the replay table in number of items.
    min_size_limiter_size: Minimum number of items required in the RB before
      sampling can begin.
  Returns:
    an instance of uniform sampling table.
  """
  rate_limiter = reverb.rate_limiters.MinSize(min_size_limiter_size)
  uniform_table = reverb.Table(
      table_name,
      max_size=table_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=rate_limiter,
      signature=data_spec)
  return uniform_table


def get_reverb_buffer(data_spec,
                      sequence_length=None,
                      table_name='uniform_table',
                      table=None,
                      reverb_server_address=None,
                      port=None,
                      replay_capacity=1000,
                      min_size_limiter_size=1):
  """Returns an instance of Reverb replay buffer and observer to add items.

  Either creates a local reverb server or uses a remote reverb server at
  reverb_sever_address (if set).

  If reverb_server_address is None, creates a local server with a uniform
  table underneath.

  Args:
    data_spec: spec of the data elements to be stored in the replay buffer
    sequence_length: integer specifying sequence_lenghts used to write
      to the given table.
    table_name: Name of the table to create.
    table: Optional table for the backing local server. If None, automatically
      creates a uniform sampling table.
    reverb_server_address: Address of the remote reverb server, if None a local
      server is created.
    port: Port to launch the server in.
    replay_capacity: Optinal (for default uniform sampling table
      only, i.e if table=None) capacity of the uniform sampling table for the
      local replay server.
    min_size_limiter_size: Optional (for default uniform sampling
      table only, i.e if table=None) minimum number of items
      required in the RB before sampling can begin, used for local server only.
  Returns:
    Reverb replay buffer instance

    Note: the if local server is created, it is not returned. It can be
      retrieved by calling local_server() on the returned replay buffer.
  """
  table_signature = tf.nest.map_structure(
      lambda s: tf.TensorSpec((sequence_length,) + s.shape, s.dtype, s.name),
      data_spec)

  if reverb_server_address is None:
    if table is None:
      table = _create_uniform_table(
          table_name,
          table_signature,
          table_capacity=replay_capacity,
          min_size_limiter_size=min_size_limiter_size)

    reverb_server = reverb.Server([table], port=port)
    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        sequence_length=sequence_length,
        table_name=table_name,
        local_server=reverb_server)
  else:
    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        sequence_length=sequence_length,
        table_name=table_name,
        server_address=reverb_server_address)

  return reverb_replay


def get_reverb_buffer_and_observer(data_spec,
                                   sequence_length=None,
                                   table_name='uniform_table',
                                   table=None,
                                   reverb_server_address=None,
                                   port=None,
                                   replay_capacity=1000,
                                   min_size_limiter_size=1,
                                   stride_length=1):
  """Returns an instance of Reverb replay buffer and observer to add items.

  Either creates a local reverb server or uses a remote reverb server at
  reverb_sever_address (if set).

  If `reverb_server_address is None`, creates a local server with a uniform
  table underneath.

  Args:
    data_spec: spec of the data elements to be stored in the replay buffer
    sequence_length: integer specifying sequence_lenghts used to write
      to the given table.
    table_name: Name of the uniform table to create.
    table: Optional table for the backing local server. If None, automatically
      creates a uniform sampling table.
    reverb_server_address: Address of the remote reverb server, if None a local
      server is created.
    port: Port to launch the server in.
    replay_capacity: capacity of the local replay server, if using (i.e. if
      reverb_server_address is None).
    min_size_limiter_size: Minimum number of items required in the RB before
      sampling can begin, used for local server only.
    stride_length: Integer strides for the sliding window for overlapping
      sequences.
  Returns:
    A tuple consisting of:
      - reverb replay buffer instance
      - replay buffer observer

    Note: the if local server is created, it is not returned. It can be
      retrieved by calling local_server() on the returned replay buffer.
  """

  reverb_replay = get_reverb_buffer(data_spec=data_spec,
                                    sequence_length=sequence_length,
                                    table_name=table_name,
                                    table=table,
                                    reverb_server_address=reverb_server_address,
                                    port=port,
                                    replay_capacity=replay_capacity,
                                    min_size_limiter_size=min_size_limiter_size)

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client, table_name, sequence_length=sequence_length,
      stride_length=stride_length)

  return reverb_replay, rb_observer
