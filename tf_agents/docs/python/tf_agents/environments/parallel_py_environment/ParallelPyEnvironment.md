<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.parallel_py_environment.ParallelPyEnvironment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="batched"/>
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="current_time_step"/>
<meta itemprop="property" content="observation_spec"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="start"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="time_step_spec"/>
</div>

# tf_agents.environments.parallel_py_environment.ParallelPyEnvironment

## Class `ParallelPyEnvironment`

Batch together environments and simulate them in external processes.

Inherits From: [`PyEnvironment`](../../../tf_agents/environments/py_environment/PyEnvironment.md)



Defined in [`environments/parallel_py_environment.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/parallel_py_environment.py).

<!-- Placeholder for "Used in" -->

The environments are created in external processes by calling the provided
callables. This can be an environment class, or a function creating the
environment and potentially wrapping it. The returned environment should not
access global variables.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Batch together environments and simulate them in external processes.

The environments can be different but must use the same action and
observation specs.

#### Args:

* <b>`env_constructors`</b>: List of callables that create environments.
* <b>`start_serially`</b>: Whether to start environments serially or in parallel.
* <b>`blocking`</b>: Whether to step environments one after another.
* <b>`flatten`</b>: Boolean, whether to use flatten action and time_steps during
    communication to reduce overhead.


#### Raises:

* <b>`ValueError`</b>: If the action or observation specs don't match.



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

<h3 id="action_spec"><code>action_spec</code></h3>

``` python
action_spec()
```



<h3 id="close"><code>close</code></h3>

``` python
close()
```

Close all external process.

<h3 id="current_time_step"><code>current_time_step</code></h3>

``` python
current_time_step()
```

Returns the current timestep.

<h3 id="observation_spec"><code>observation_spec</code></h3>

``` python
observation_spec()
```



<h3 id="render"><code>render</code></h3>

``` python
render(mode='rgb_array')
```

Renders the environment.

#### Args:

* <b>`mode`</b>: One of ['rgb_array', 'human']. Renders to an numpy array, or brings
    up a window where the environment can be visualized.

#### Returns:

An ndarray of shape [width, height, 3] denoting an RGB image if mode is
`rgb_array`. Otherwise return nothing and render directly to a display
window.

#### Raises:

* <b>`NotImplementedError`</b>: If the environment does not support rendering.

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

<h3 id="start"><code>start</code></h3>

``` python
start()
```



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





