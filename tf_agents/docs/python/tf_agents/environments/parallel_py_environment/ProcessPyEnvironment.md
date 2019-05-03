<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.parallel_py_environment.ProcessPyEnvironment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="observation_spec"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="start"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="time_step_spec"/>
<meta itemprop="property" content="wait_start"/>
</div>

# tf_agents.environments.parallel_py_environment.ProcessPyEnvironment

## Class `ProcessPyEnvironment`

Step a single env in a separate process for lock free paralellism.





Defined in [`environments/parallel_py_environment.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/parallel_py_environment.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    env_constructor,
    flatten=False
)
```

Step environment in a separate process for lock free paralellism.

The environment is created in an external process by calling the provided
callable. This can be an environment class, or a function creating the
environment and potentially wrapping it. The returned environment should
not access global variables.

#### Args:

* <b>`env_constructor`</b>: Callable that creates and returns a Python environment.
* <b>`flatten`</b>: Boolean, whether to assume flattened actions and time_steps
    during communication to avoid overhead.


#### Attributes:

* <b>`observation_spec`</b>: The cached observation spec of the environment.
* <b>`action_spec`</b>: The cached action spec of the environment.
* <b>`time_step_spec`</b>: The cached time step spec of the environment.



## Methods

<h3 id="__getattr__"><code>__getattr__</code></h3>

``` python
__getattr__(name)
```

Request an attribute from the environment.

Note that this involves communication with the external process, so it can
be slow.

#### Args:

* <b>`name`</b>: Attribute to access.


#### Returns:

Value of the attribute.

<h3 id="action_spec"><code>action_spec</code></h3>

``` python
action_spec()
```



<h3 id="call"><code>call</code></h3>

``` python
call(
    name,
    *args,
    **kwargs
)
```

Asynchronously call a method of the external environment.

#### Args:

* <b>`name`</b>: Name of the method to call.
* <b>`*args`</b>: Positional arguments to forward to the method.
* <b>`**kwargs`</b>: Keyword arguments to forward to the method.


#### Returns:

Promise object that blocks and provides the return value when called.

<h3 id="close"><code>close</code></h3>

``` python
close()
```

Send a close message to the external process and join it.

<h3 id="observation_spec"><code>observation_spec</code></h3>

``` python
observation_spec()
```



<h3 id="reset"><code>reset</code></h3>

``` python
reset(blocking=True)
```

Reset the environment.

#### Args:

* <b>`blocking`</b>: Whether to wait for the result.


#### Returns:

New observation when blocking, otherwise callable that returns the new
observation.

<h3 id="start"><code>start</code></h3>

``` python
start(wait_to_start=True)
```

Start the process.

#### Args:

* <b>`wait_to_start`</b>: Whether the call should wait for an env initialization.

<h3 id="step"><code>step</code></h3>

``` python
step(
    action,
    blocking=True
)
```

Step the environment.

#### Args:

* <b>`action`</b>: The action to apply to the environment.
* <b>`blocking`</b>: Whether to wait for the result.


#### Returns:

time step when blocking, otherwise callable that returns the time step.

<h3 id="time_step_spec"><code>time_step_spec</code></h3>

``` python
time_step_spec()
```



<h3 id="wait_start"><code>wait_start</code></h3>

``` python
wait_start()
```

Wait for the started process to finish initialization.



