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

# Lint as: python2, python3
"""Helper functions for running models in a distributed setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def get_distribution_strategy(distribution_strategy="default",
                              num_gpus=0,
                              num_packs=-1):
  """Return a DistributionStrategy for running the model.

  Args:
    distribution_strategy: a string specifying which distribution strategy to
      use. Accepted values are 'off', 'default', 'one_device', and 'mirrored'
      case insensitive. 'off' means not to use Distribution Strategy; 'default'
      means to choose from `MirroredStrategy`or `OneDeviceStrategy` according to
      the number of GPUs.
    num_gpus: Number of GPUs to run this model.
    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`.

  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is 'off' or 'one_device' and
      `num_gpus` is larger than 1; or `num_gpus` is negative.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  distribution_strategy = distribution_strategy.lower()
  if distribution_strategy == "off":
    if num_gpus > 1:
      raise ValueError("When {} GPUs are specified, distribution_strategy "
                       "cannot be set to 'off'.".format(num_gpus))
    return None

  if (distribution_strategy == "one_device" or
      (distribution_strategy == "default" and num_gpus <= 1)):
    if num_gpus == 0:
      return tf.distribute.OneDeviceStrategy("device:CPU:0")
    else:
      if num_gpus > 1:
        raise ValueError("`OneDeviceStrategy` can not be used for more than "
                         "one device.")
      return tf.distribute.OneDeviceStrategy("device:GPU:0")

  if distribution_strategy in ("mirrored", "default"):
    if num_gpus == 0:
      assert distribution_strategy == "mirrored"
      devices = ["device:CPU:0"]
    else:
      devices = ["device:GPU:%d" % i for i in range(num_gpus)]

    cross_device_ops = None
    if num_packs > -1:
      cross_device_ops = tf.distribute.NcclAllReduce(num_packs=num_packs)
    return tf.distribute.MirroredStrategy(devices=devices,
                                          cross_device_ops=cross_device_ops)


def strategy_scope_context(strategy):
  if strategy:
    strategy_scope = strategy.scope()
  else:
    strategy_scope = DummyContextManager()

  return strategy_scope


class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass
