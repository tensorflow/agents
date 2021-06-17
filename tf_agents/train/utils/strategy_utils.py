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
"""Utilities for managing distrubtion strategies."""
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

flags.DEFINE_string('tpu', None, 'BNS address for the TPU')
flags.DEFINE_bool('use_gpu', False, 'If True a MirroredStrategy will be used.')


def get_strategy(tpu, use_gpu):
  """Utility to create a `tf.DistributionStrategy` for TPU or GPU.

  If neither is being used a DefaultStrategy is returned which allows executing
  on CPU only.

  Args:
    tpu: BNS address of TPU to use. Note the flag and param are called TPU as
      that is what the xmanager utilities call.
    use_gpu: Whether a GPU should be used. This will create a MirroredStrategy.

  Raises:
    ValueError if both tpu and use_gpu are set.
  Returns:
    An instance of a `tf.DistributionStrategy`.
  """
  if tpu and use_gpu:
    raise ValueError('Only one of tpu or use_gpu should be provided.')
  if tpu or use_gpu:
    logging.info('Devices: \n%s', tf.config.list_logical_devices())
    if tpu:
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)

      strategy = tf.distribute.TPUStrategy(resolver)
    else:
      strategy = tf.distribute.MirroredStrategy()
    logging.info('Devices after getting strategy:\n%s',
                 tf.config.list_logical_devices())
  else:
    strategy = tf.distribute.get_strategy()

  return strategy
