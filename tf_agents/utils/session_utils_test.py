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

"""Tests for learning.reinforment_learning.utils.session_user."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.utils import session_utils


class MySessionUser(session_utils.SessionUser):

  def __init__(self):
    super(MySessionUser, self).__init__()
    self._op = tf.constant(0)

  def run(self):
    return self.session.run(self._op)


class TfRunnableTest(tf.test.TestCase):

  def setUp(self):
    if tf.executing_eagerly():
      self.skipTest("session_utils are not applicable when executing eagerly")

  def testWithoutSession(self):
    session_user = MySessionUser()
    with self.assertRaisesRegexp(AttributeError, "No TensorFlow session"):
      session_user.run()

  def testSessionWithinContextManager(self):
    session_user = MySessionUser()
    with tf.compat.v1.Session() as session:
      self.assertIs(session_user.session, session)
      self.assertEqual(0, session_user.run())

  def testTestSessionWithinContextManager(self):
    session_user = MySessionUser()
    with self.cached_session() as session:
      self.assertIs(session_user.session, session)
      self.assertEqual(0, session_user.run())

  def testSessionWithinMonitoredSessionContextManagerRaisesError(self):
    session_user = MySessionUser()
    with tf.compat.v1.train.MonitoredSession() as _:
      with self.assertRaisesRegexp(AttributeError, "No TensorFlow session"):
        session_user.run()

  def testSessionWithSingularMonitoredSession(self):
    session_user = MySessionUser()
    with tf.compat.v1.train.SingularMonitoredSession() as session:
      session_user.session = session
      self.assertEqual(0, session_user.run())

  def testSessionWithMonitoredSession(self):
    session_user = MySessionUser()
    with tf.compat.v1.train.MonitoredSession() as session:
      session_user.session = session
      self.assertEqual(0, session_user.run())

  def testSessionProvidedUsingSetSession(self):
    session_user = MySessionUser()
    session = tf.compat.v1.Session()
    session_user.session = session
    self.assertIs(session_user.session, session)
    self.assertEqual(0, session_user.run())

  def testSettingSessionTakesPrecedenceOverDefaultSession(self):
    session_user = MySessionUser()
    with self.cached_session() as test_session:
      session = tf.compat.v1.Session()
      self.assertIsNot(test_session, session)
      self.assertIs(session_user.session, test_session)
      session_user.session = session
      self.assertIs(session_user.session, session)

  def testSessionUsesCurrent(self):
    session_user = MySessionUser()
    session1 = tf.compat.v1.Session()
    session2 = tf.compat.v1.Session()
    self.assertIsNot(session1, session2)
    with session1.as_default():
      self.assertIs(session_user.session, session1)
    with session2.as_default():
      self.assertIs(session_user.session, session2)
    session1.close()
    session2.close()


if __name__ == "__main__":
  tf.test.main()
