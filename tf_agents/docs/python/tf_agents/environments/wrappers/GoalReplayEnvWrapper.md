<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.wrappers.GoalReplayEnvWrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="batched"/>
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__getattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="current_time_step"/>
<meta itemprop="property" content="get_goal_from_trajectory"/>
<meta itemprop="property" content="get_trajectory_with_goal"/>
<meta itemprop="property" content="observation_spec"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="time_step_spec"/>
<meta itemprop="property" content="wrapped_env"/>
</div>

# tf_agents.environments.wrappers.GoalReplayEnvWrapper

## Class `GoalReplayEnvWrapper`

Adds a goal to the observation, used for HER (Hindsight Experience Replay).

Inherits From: [`PyEnvironmentBaseWrapper`](../../../tf_agents/environments/wrappers/PyEnvironmentBaseWrapper.md)



Defined in [`environments/wrappers.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/wrappers.py).

<!-- Placeholder for "Used in" -->

Sources:
  [1] Hindsight Experience Replay. https://arxiv.org/abs/1707.01495.

To use this wrapper, create an environment-specific version by inheriting this
class.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(env)
```

Initializes a wrapper to add a goal to the observation.

#### Args:

* <b>`env`</b>: A `py_environment.PyEnvironment` environment to wrap.


#### Raises:

* <b>`ValueError`</b>: If environment observation is not a dict



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>



<h3 id="batched"><code>batched</code></h3>





## Methods

<h3 id="__enter__"><code>__enter__</code></h3>

``` python
__enter__()
```

Allows the environment to be used in a with-statement context.

<h3 id="__exit__"><code>__exit__</code></h3>

``` python
__exit__(
    unused_exception_type,
    unused_exc_value,
    unused_traceback
)
```

Allows the environment to be used in a with-statement context.

<h3 id="__getattr__"><code>__getattr__</code></h3>

``` python
__getattr__(name)
```

Forward all other calls to the base environment.

<h3 id="action_spec"><code>action_spec</code></h3>

``` python
action_spec()
```



<h3 id="close"><code>close</code></h3>

``` python
close()
```

Frees any resources used by the environment.

Implement this method for an environment backed by an external process.

This method be used directly

```python
env = Env(...)
# Use env.
env.close()
```

or via a context manager

```python
with Env(...) as env:
  # Use env.
```

<h3 id="current_time_step"><code>current_time_step</code></h3>

``` python
current_time_step()
```

Returns the current timestep.

<h3 id="get_goal_from_trajectory"><code>get_goal_from_trajectory</code></h3>

``` python
get_goal_from_trajectory(trajectory)
```

Extracts the goal from a given trajectory.

#### Args:

* <b>`trajectory`</b>: An instance of `Trajectory`.


#### Returns:

Environment specific goal


#### Raises:

* <b>`NotImplementedError`</b>: function should be implemented in child class.

<h3 id="get_trajectory_with_goal"><code>get_trajectory_with_goal</code></h3>

``` python
get_trajectory_with_goal(
    trajectory,
    goal
)
```

Generates a new trajectory assuming the given goal was the actual target.

One example is updating a "distance-to-goal" field in the observation. Note
that relevant state information must be recovered or re-calculated from the
given trajectory.

#### Args:

* <b>`trajectory`</b>: An instance of `Trajectory`.
* <b>`goal`</b>: Environment specific goal


#### Returns:

Updated instance of `Trajectory`


#### Raises:

* <b>`NotImplementedError`</b>: function should be implemented in child class.

<h3 id="observation_spec"><code>observation_spec</code></h3>

``` python
observation_spec()
```



<h3 id="render"><code>render</code></h3>

``` python
render(mode='rgb_array')
```



<h3 id="reset"><code>reset</code></h3>

``` python
reset()
```

Starts a new sequence and returns the first `TimeStep` of this sequence.

Note: Subclasses cannot override this directly. Subclasses implement
_reset() which will be called by this method. The output of _reset() will
be cached and made available through current_time_step().

#### Returns:

A `TimeStep` namedtuple containing:
* <b>`step_type`</b>: A `StepType` of `FIRST`.
* <b>`reward`</b>: 0.0, indicating the reward.
* <b>`discount`</b>: 1.0, indicating the discount.
* <b>`observation`</b>: A NumPy array, or a nested dict, list or tuple of arrays
      corresponding to `observation_spec()`.

<h3 id="step"><code>step</code></h3>

``` python
step(action)
```

Updates the environment according to the action and returns a `TimeStep`.

If the environment returned a `TimeStep` with `StepType.LAST` at the
previous step, this call to `step` will reset the environment,
start a new sequence and `action` will be ignored.

This method will also start a new sequence if called after the environment
has been constructed and `reset` has not been called. Again, in this case
`action` will be ignored.

Note: Subclasses cannot override this directly. Subclasses implement
_step() which will be called by this method. The output of _step() will be
cached and made available through current_time_step().

#### Args:

* <b>`action`</b>: A NumPy array, or a nested dict, list or tuple of arrays
    corresponding to `action_spec()`.


#### Returns:

A `TimeStep` namedtuple containing:
* <b>`step_type`</b>: A `StepType` value.
* <b>`reward`</b>: A NumPy array, reward value for this timestep.
* <b>`discount`</b>: A NumPy array, discount in the range [0, 1].
* <b>`observation`</b>: A NumPy array, or a nested dict, list or tuple of arrays
      corresponding to `observation_spec()`.

<h3 id="time_step_spec"><code>time_step_spec</code></h3>

``` python
time_step_spec()
```

Describes the `TimeStep` fields returned by `step()`.

Override this method to define an environment that uses non-standard values
for any of the items returned by `step()`. For example, an environment with
array-valued rewards.

#### Returns:

A `TimeStep` namedtuple containing (possibly nested) `ArraySpec`s defining
the step_type, reward, discount, and observation structure.

<h3 id="wrapped_env"><code>wrapped_env</code></h3>

``` python
wrapped_env()
```





