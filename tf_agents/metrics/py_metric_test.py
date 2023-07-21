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

"""Tests for tf_agents.metrics.py_metric."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import py_metric


class DummyMetric(py_metric.PyStepMetric):

  def __init__(self, name='Metric'):
    super(DummyMetric, self).__init__(name)
    self.reset()

  def reset(self):
    self.value = 0

  def result(self):
    return self.value

  def call(self, trajectory):
    pass


class PyMetricSummariesTest(tf.test.TestCase):

  def setUp(self):
    super(PyMetricSummariesTest, self).setUp()
    self.summary_dir = tempfile.mkdtemp(dir=os.getenv('TEST_TMPDIR'))
    self.writer = tf.compat.v2.summary.create_file_writer(self.summary_dir)
    self.writer.set_as_default()
    self.metric1 = DummyMetric('Metric1')
    self.metric2 = DummyMetric('Metric2')
    self.metric3 = DummyMetric('Metric3')
    self.global_step = tf.compat.v1.train.get_or_create_global_step()
    self.incr_global_step = tf.compat.v1.assign_add(self.global_step, 1)

  def testBuildsSummary(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881100')
    metric = DummyMetric()
    self.assertIsNone(metric.summary_op)
    metric.tf_summaries(train_step=self.global_step)
    self.assertIsNotNone(metric.summary_op)

  def assert_summary_equals(self, records, tag, step, value):
    for record in records[1:]:
      if record.summary.value[0].tag != tag:
        continue
      if record.step != step:
        continue
      self.assertEqual(value, tf.make_ndarray(record.summary.value[0].tensor))
      return
    self.fail(
        'Could not find record for tag {} and step {}'.format(tag, step))

  def get_records(self):
    files = os.listdir(self.summary_dir)
    self.assertEqual(1, len(files))
    file_path = os.path.join(self.summary_dir, files[0])
    return list(tf.compat.v1.train.summary_iterator(file_path))

  def testSummarySimple(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881100')
    with tf.compat.v2.summary.record_if(True):
      self.metric1.tf_summaries(train_step=self.global_step)
      self.metric2.tf_summaries(train_step=self.global_step)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(self.writer.init())
      self.metric1.value = 3
      py_metric.run_summaries([self.metric1, self.metric2])
      sess.run(self.writer.flush())

    records = self.get_records()

    # 2 summaries + 1 file header
    self.assertEqual(3, len(records))

    self.assert_summary_equals(records, 'Metrics/Metric1', 0, 3)
    self.assert_summary_equals(records, 'Metrics/Metric2', 0, 0)

  def testSummaryUpdates(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881100')
    with tf.compat.v2.summary.record_if(True):
      self.metric1.tf_summaries(train_step=self.global_step)
      self.metric2.tf_summaries(train_step=self.global_step)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(self.writer.init())
      self.metric1.value = 3
      self.metric2.value = 0
      py_metric.run_summaries([self.metric1, self.metric2])
      sess.run(self.incr_global_step)
      self.metric1.value = 5
      self.metric2.value = 2
      py_metric.run_summaries([self.metric1, self.metric2])
      sess.run(self.writer.flush())

    records = self.get_records()

    # 4 summaries + 1 file header
    self.assertEqual(5, len(records))

    self.assert_summary_equals(records, 'Metrics/Metric1', 0, 3)
    self.assert_summary_equals(records, 'Metrics/Metric2', 0, 0)
    self.assert_summary_equals(records, 'Metrics/Metric1', 1, 5)
    self.assert_summary_equals(records, 'Metrics/Metric2', 1, 2)

  def testSummaryStepMetrics(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881100')
    with tf.compat.v2.summary.record_if(True):
      self.metric1.tf_summaries(
          train_step=self.global_step, step_metrics=(self.metric2,))
      self.metric2.tf_summaries(
          train_step=self.global_step, step_metrics=(self.metric2,))

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(self.writer.init())
      self.metric1.value = 3
      self.metric2.value = 2
      py_metric.run_summaries([self.metric1, self.metric2])
      sess.run(self.writer.flush())

    records = self.get_records()

    # (2 records for metric1, 1 for metric2) + 1 file header
    self.assertEqual(4, len(records))

    self.assert_summary_equals(records, 'Metrics/Metric1', 0, 3)
    self.assert_summary_equals(records, 'Metrics_vs_Metric2/Metric1', 2, 3)

    self.assert_summary_equals(records, 'Metrics/Metric2', 0, 2)

  def testSummaryStepMetricsUpdate(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881100')
    with tf.compat.v2.summary.record_if(True):
      self.metric1.tf_summaries(
          train_step=self.global_step, step_metrics=(self.metric2,))
      self.metric2.tf_summaries(
          train_step=self.global_step, step_metrics=(self.metric2,))

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(self.writer.init())
      self.metric1.value = 3
      self.metric2.value = 2
      py_metric.run_summaries([self.metric1, self.metric2])
      self.metric1.value = 4
      self.metric2.value = 3
      py_metric.run_summaries([self.metric1, self.metric2])
      sess.run(self.writer.flush())

    records = self.get_records()

    # (2 records for metric1, 1 for metric2) * 2 + 1 file header
    self.assertEqual(7, len(records))

    self.assert_summary_equals(records, 'Metrics_vs_Metric2/Metric1', 2, 3)
    self.assert_summary_equals(records, 'Metrics_vs_Metric2/Metric1', 3, 4)

  def testSummaryMultipleStepMetrics(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881100')
    with tf.compat.v2.summary.record_if(True):
      self.metric1.tf_summaries(
          train_step=self.global_step,
          step_metrics=(self.metric2, self.metric3))
      self.metric2.tf_summaries(
          train_step=self.global_step,
          step_metrics=(self.metric2, self.metric3))
      self.metric3.tf_summaries(
          train_step=self.global_step,
          step_metrics=(self.metric2, self.metric3))

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(self.writer.init())
      self.metric1.value = 1
      self.metric2.value = 2
      self.metric3.value = 3
      py_metric.run_summaries([self.metric1, self.metric2, self.metric3])
      sess.run(self.writer.flush())

    records = self.get_records()

    # (3 records for metric1, 2 for metric2, 2 for metric3) + 1 file header
    self.assertEqual(8, len(records))

    self.assert_summary_equals(records, 'Metrics/Metric1', 0, 1)
    self.assert_summary_equals(records, 'Metrics_vs_Metric2/Metric1', 2, 1)
    self.assert_summary_equals(records, 'Metrics_vs_Metric3/Metric1', 3, 1)

    self.assert_summary_equals(records, 'Metrics/Metric2', 0, 2)
    self.assert_summary_equals(records, 'Metrics_vs_Metric3/Metric2', 3, 2)

    self.assert_summary_equals(records, 'Metrics/Metric3', 0, 3)
    self.assert_summary_equals(records, 'Metrics_vs_Metric2/Metric3', 2, 3)


if __name__ == '__main__':
  tf.test.main()
