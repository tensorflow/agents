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

"""Base class for TensorFlow metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.utils import common as common_utils
from tensorflow.contrib.eager.python import metrics as eager_metrics  # TF internal


class TFStepMetric(eager_metrics.Metric):
  """Defines the interface for TF metrics."""

  def __init__(self, name, prefix='Metrics', **kwargs):
    super(TFStepMetric, self).__init__(name, **kwargs)
    self._prefix = prefix

  def tf_summaries(self, step_metrics=()):
    prefix = self._prefix
    tag = common_utils.join_scope(prefix, self.name)
    result = self.result()
    tf.contrib.summary.scalar(name=tag, tensor=result)
    if prefix:
      prefix += '_'
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
      step = step_metric.result()
      tf.contrib.summary.scalar(
          name=step_tag,
          tensor=result,
          step=step)

