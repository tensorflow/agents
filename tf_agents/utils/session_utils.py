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

"""A class to create objects which needs a session to be functional.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SessionUser(object):
  """A class which needs a TensorFlow session for some of its operations.

  A `SessionUser` is a class which can be instantiated outside of a
  TensorFlow session, but which needs to have access to a session-like object
  for most of its operations.

  A session-like object is an object on which we can call `run` to execute
  some TensorFlow ops (e.g. `tf.Session()` and
  `tf.train.(Singular)MonitoredSession`).

  There are 2 ways of providing a session to a `SessionUser`:
  - within a TensorFlow session context manager (e.g. within
    `with tf.Session() as session:`), the session will be automatically
    retrieved. Be aware that a `tf.train.(Singular)MonitoredSession` does not
    enter a session context manager.
  - if the session is constructed outside of a context manager, it must be
    provided using the `session` setter.

  The session can then be accessed using the `session` property.

  The usual way to use a `SessionUser` is the following.
  ```python
  class MySessionUserClass(SessionUser):

    def __init__(self):
      self(MySessionUserClass, self).__init__()
      self.op = tf.constant(0)

    def run_some_op(self):
      self.session.run(self.op)

  my_session_owner = MySessionUserClass()
  with tf.Session() as session:
    my_session_owner.run_some_op()
  ```

  Since both `tf.train.SingularMonitoredSession` and `tf.train.MonitoredSession`
  do not create a Session context manager, one will need to set the session
  manually.
  ```python
  with tf.train.(Singular)MonitoredSession(...) as session:
    my_session_owner.session = session
    my_session_owner.run_some_op()
  ```

  For `tf.train.SingularMonitoredSession`, since one can access the
  underlying raw session, one can also open a Session context manager.
  ```python
  with tf.train.SingularMonitoredSession(...) as mon_sess:
    with mon_sess.raw_session().as_default():
       while not mon_sess.should_stop():
         my_session_owner.run_some_op()
  ```

  Advanced usage:

  One can override the session setter by using the following code.
  ```python


  class MyClass(session_utils.SessionUser):

    # This is overriding the `session` setter from `session_utils.SessionUser`.
    @session_utils.SessionUser.session.setter
    def session(self, session):
      # This calls the setter of the `session_utils.SessionUser` class.
      session_utils.SessionUser.session.fset(self, session)
      # Then you can do other things such as setting the session of internal
      # objects.
  ```
  """

  @property
  def session(self):
    """Returns the TensorFlow session-like object used by this object.

    Returns:
      The internal TensorFlow session-like object. If it is `None`, it will
      return the current TensorFlow session context manager.

    Raises:
      AttributeError: When no session-like object has been set, and no
        session context manager has been entered.
    """
    if not hasattr(self, "_session_user_internal_session"):
      self._session_user_internal_session = None

    if self._session_user_internal_session is not None:
      return self._session_user_internal_session

    default_session = tf.compat.v1.get_default_session()
    if default_session is None:
      raise AttributeError(
          "No TensorFlow session-like object was set on this {!r}, and none "
          "could be retrieved using 'tf.get_default_session()'.".format(
              self.__class__.__name__))
    return default_session

  @session.setter
  def session(self, session):
    """Sets up the internal session-like object."""
    self._session_user_internal_session = session
