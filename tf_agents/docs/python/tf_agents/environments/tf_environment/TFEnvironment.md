<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.tf_environment.TFEnvironment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="batched"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="current_time_step"/>
<meta itemprop="property" content="observation_spec"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="time_step_spec"/>
</div>

# tf_agents.environments.tf_environment.TFEnvironment

## Class `TFEnvironment`

Abstract base class for TF RL environments.





Defined in [`environments/tf_environment.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py).

<!-- Placeholder for "Used in" -->

The `current_time_step()` method returns current `time_step`, resetting the
environment if necessary.

The `step(action)` method applies the action and returns the new `time_step`.
This method will also reset the environment if needed and ignore the action in
that case.

The `reset()` method returns `time_step` that results from an environment
reset and is guaranteed to have step_type=ts.FIRST

The `reset()` method is only needed for explicit resets. In general, the
environment will reset automatically when needed, for example, when no
episode was started or when it reaches a step after the end of the episode
(i.e. step_type=ts.LAST).

Example for collecting an episode in eager mode:

  tf_env = TFEnvironment()

  # reset() creates the initial time_step and resets the environment.
  time_step = tf_env.reset()
  while not time_step.is_last():
    action_step = policy.action(time_step)
    time_step = tf_env.step(action_step.action)

Example of simple use in graph mode:

  tf_env = TFEnvironment()

  # current_time_step() creates the initial TimeStep.
  time_step = tf_env.current_time_step()
  action_step = policy.action(time_step)
  # Apply the action and return the new TimeStep.
  next_time_step = tf_env.step(action_step.action)

  sess.run([time_step, action_step, next_time_step])

Example with explicit resets in graph mode:

  reset_op = tf_env.reset()
  time_step = tf_env.current_time_step()
  action_step = policy.action(time_step)
  # Apply the action and return the new TimeStep.
  next_time_step = tf_env.step(action_step.action)

  # The environment will initialize before starting.
  sess.run([time_step, action_step, next_time_step])
  # This will force reset the Environment.
  sess.run(reset_op)
  # This will apply a new action in the environment.
  sess.run([time_step, action_step, next_time_step])

Example of random actions in graph mode:

  tf_env = TFEnvironment()

  # Action needs to depend on the time_step using control_dependencies.
  time_step = tf_env.current_time_step()
  with tf.control_dependencies([time_step.step_type]):
    action = tensor_spec.sample_bounded_spec(tf_env.action_spec())
  next_time_step = tf_env.step(action)

  sess.run([time_step, action, next_time_step])

Example of collecting full episodes with a while_loop:

  tf_env = TFEnvironment()

  # reset() creates the initial time_step
  time_step = tf_env.reset()
  c = lambda t: tf.logical_not(t.is_last())
  body = lambda t: [tf_env.step(t.observation)]

  final_time_step = tf.while_loop(c, body, [time_step])

  sess.run(final_time_step)

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    time_step_spec=None,
    action_spec=None,
    batch_size=1
)
```

Initializes the environment.

Meant to be called by subclass constructors.

#### Args:

* <b>`time_step_spec`</b>: A `TimeStep` namedtuple containing `TensorSpec`s
    defining the Tensors returned by
    `step()` (step_type, reward, discount, and observation).
* <b>`action_spec`</b>: A nest of BoundedTensorSpec representing the actions of the
    environment.
* <b>`batch_size`</b>: The batch size expected for the actions and observations.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>



<h3 id="batched"><code>batched</code></h3>





## Methods

<h3 id="action_spec"><code>action_spec</code></h3>

``` python
action_spec()
```

Describes the specs of the Tensors expected by `step(action)`.

`action` can be a single Tensor, or a nested dict, list or tuple of
Tensors.

#### Returns:

An single `TensorSpec`, or a nested dict, list or tuple of
`TensorSpec` objects, which describe the shape and
dtype of each Tensor expected by `step()`.

<h3 id="current_time_step"><code>current_time_step</code></h3>

``` python
current_time_step()
```

Returns the current `TimeStep`.

#### Returns:

A `TimeStep` namedtuple containing:
* <b>`step_type`</b>: A `StepType` value.
* <b>`reward`</b>: Reward at this time_step.
* <b>`discount`</b>: A discount in the range [0, 1].
* <b>`observation`</b>: A Tensor, or a nested dict, list or tuple of Tensors
      corresponding to `observation_spec()`.

<h3 id="observation_spec"><code>observation_spec</code></h3>

``` python
observation_spec()
```

Defines the `TensorSpec` of observations provided by the environment.

#### Returns:

A `TensorSpec`, or a nested dict, list or tuple of
`TensorSpec` objects, which describe the observation.

<h3 id="render"><code>render</code></h3>

``` python
render()
```

Renders a frame from the environment.

#### Raises:

* <b>`NotImplementedError`</b>: If the environment does not support rendering.

<h3 id="reset"><code>reset</code></h3>

``` python
reset()
```

Resets the environment and returns the current time_step.

#### Returns:

A `TimeStep` namedtuple containing:
* <b>`step_type`</b>: A `StepType` value.
* <b>`reward`</b>: Reward at this time_step.
* <b>`discount`</b>: A discount in the range [0, 1].
* <b>`observation`</b>: A Tensor, or a nested dict, list or tuple of Tensors
      corresponding to `observation_spec()`.

<h3 id="step"><code>step</code></h3>

``` python
step(action)
```

Steps the environment according to the action.

If the environment returned a `TimeStep` with `StepType.LAST` at the
previous step, this call to `step` should reset the environment (note that
it is expected that whoever defines this method, calls reset in this case),
start a new sequence and `action` will be ignored.

This method will also start a new sequence if called after the environment
has been constructed and `reset()` has not been called. In this case
`action` will be ignored.

Expected sequences look like:

  time_step -> action -> next_time_step

The action should depend on the previous time_step for correctness.

#### Args:

* <b>`action`</b>: A Tensor, or a nested dict, list or tuple of Tensors
    corresponding to `action_spec()`.


#### Returns:

A `TimeStep` namedtuple containing:
* <b>`step_type`</b>: A `StepType` value.
* <b>`reward`</b>: Reward at this time_step.
* <b>`discount`</b>: A discount in the range [0, 1].
* <b>`observation`</b>: A Tensor, or a nested dict, list or tuple of Tensors
      corresponding to `observation_spec()`.

<h3 id="time_step_spec"><code>time_step_spec</code></h3>

``` python
time_step_spec()
```

Describes the `TimeStep` specs of Tensors returned by `step()`.

#### Returns:

A `TimeStep` namedtuple containing `TensorSpec` objects defining the
Tensors returned by `step()`, i.e.
(step_type, reward, discount, observation).



