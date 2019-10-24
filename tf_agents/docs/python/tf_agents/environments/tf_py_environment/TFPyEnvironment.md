<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.tf_py_environment.TFPyEnvironment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="batched"/>
<meta itemprop="property" content="pyenv"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="current_time_step"/>
<meta itemprop="property" content="observation_spec"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="time_step_spec"/>
</div>

# tf_agents.environments.tf_py_environment.TFPyEnvironment

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_py_environment.py">View
source</a>

## Class `TFPyEnvironment`

Exposes a Python environment as an in-graph TF environment.

Inherits From: [`TFEnvironment`](../../../tf_agents/environments/tf_environment/TFEnvironment.md)

<!-- Placeholder for "Used in" -->

This class supports Python environments that return nests of arrays as
observations and accept nests of arrays as actions. The nest structure is
reflected in the in-graph environment's observation and action structure.

#### Implementation notes:

* Since `tf.py_func` deals in lists of tensors, this class has some additional
  `tf.nest.flatten` and `tf.nest.pack_structure_as` calls.

* This class currently cast rewards and discount to float32.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Initializes a new `TFPyEnvironment`.

#### Args:

*   <b>`environment`</b>: Environment to interact with, implementing
    <a href="../../../tf_agents/environments/py_environment/PyEnvironment.md"><code>py_environment.PyEnvironment</code></a>.
*   <b>`check_dims`</b>: Whether should check batch dimensions of actions in
    `step`.

#### Raises:

*   <b>`TypeError`</b>: If `environment` is not a subclass of
    <a href="../../../tf_agents/environments/py_environment/PyEnvironment.md"><code>py_environment.PyEnvironment</code></a>.

## Properties

<h3 id="batch_size"><code>batch_size</code></h3>

<h3 id="batched"><code>batched</code></h3>

<h3 id="pyenv"><code>pyenv</code></h3>

Returns the underlying Python environment.

## Methods

<h3 id="action_spec"><code>action_spec</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

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

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

``` python
current_time_step()
```

Returns the current `TimeStep`.

#### Returns:

A `TimeStep` namedtuple containing: step_type: A `StepType` value. reward:
Reward at this time_step. discount: A discount in the range [0, 1]. observation:
A Tensor, or a nested dict, list or tuple of Tensors corresponding to
`observation_spec()`.

<h3 id="observation_spec"><code>observation_spec</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

``` python
observation_spec()
```

Defines the `TensorSpec` of observations provided by the environment.

#### Returns:

A `TensorSpec`, or a nested dict, list or tuple of
`TensorSpec` objects, which describe the observation.

<h3 id="render"><code>render</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

``` python
render()
```

Renders a frame from the environment.

#### Raises:

* <b>`NotImplementedError`</b>: If the environment does not support rendering.

<h3 id="reset"><code>reset</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

``` python
reset()
```

Resets the environment and returns the current time_step.

#### Returns:

A `TimeStep` namedtuple containing: step_type: A `StepType` value. reward:
Reward at this time_step. discount: A discount in the range [0, 1]. observation:
A Tensor, or a nested dict, list or tuple of Tensors corresponding to
`observation_spec()`.

<h3 id="step"><code>step</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

``` python
step(action)
```

Steps the environment according to the action.

If the environment returned a `TimeStep` with
<a href="../../../tf_agents/trajectories/time_step/StepType.md#LAST"><code>StepType.LAST</code></a>
at the previous step, this call to `step` should reset the environment (note
that it is expected that whoever defines this method, calls reset in this case),
start a new sequence and `action` will be ignored.

This method will also start a new sequence if called after the environment
has been constructed and `reset()` has not been called. In this case
`action` will be ignored.

Expected sequences look like:

  time_step -> action -> next_time_step

The action should depend on the previous time_step for correctness.

#### Args:

*   <b>`action`</b>: A Tensor, or a nested dict, list or tuple of Tensors
    corresponding to `action_spec()`.

#### Returns:

A `TimeStep` namedtuple containing: step_type: A `StepType` value. reward:
Reward at this time_step. discount: A discount in the range [0, 1]. observation:
A Tensor, or a nested dict, list or tuple of Tensors corresponding to
`observation_spec()`.

<h3 id="time_step_spec"><code>time_step_spec</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

``` python
time_step_spec()
```

Describes the `TimeStep` specs of Tensors returned by `step()`.

#### Returns:

A `TimeStep` namedtuple containing `TensorSpec` objects defining the
Tensors returned by `step()`, i.e.
(step_type, reward, discount, observation).
