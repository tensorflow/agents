<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.session_utils.SessionUser" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="session"/>
</div>

# tf_agents.utils.session_utils.SessionUser

## Class `SessionUser`

A class which needs a TensorFlow session for some of its operations.





Defined in [`utils/session_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/session_utils.py).

<!-- Placeholder for "Used in" -->

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

## Properties

<h3 id="session"><code>session</code></h3>

Returns the TensorFlow session-like object used by this object.

#### Returns:

The internal TensorFlow session-like object. If it is `None`, it will
return the current TensorFlow session context manager.


#### Raises:

* <b>`AttributeError`</b>: When no session-like object has been set, and no
    session context manager has been entered.



