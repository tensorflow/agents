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

"""Utils to export metrics."""

from absl import logging


def export_metrics(step, metrics, loss_info=None):
  """Exports the metrics and loss information to logging.info.

  Args:
    step: Integer denoting the round at which we log the metrics.
    metrics: List of `TF metrics` to log.
    loss_info: An optional instance of `LossInfo` whose value is logged.
  """
  def logging_at_step_fn(name, value):
    logging_msg = f'[step={step}] {name} = {value}.'
    logging.info(logging_msg)

  for metric in metrics:
    logging_at_step_fn(metric.name, metric.result())
  if loss_info is not None:
    logging_at_step_fn('loss', loss_info.loss)
