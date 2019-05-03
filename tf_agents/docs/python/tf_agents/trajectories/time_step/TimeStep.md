<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step.TimeStep" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="step_type"/>
<meta itemprop="property" content="reward"/>
<meta itemprop="property" content="discount"/>
<meta itemprop="property" content="observation"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="is_first"/>
<meta itemprop="property" content="is_last"/>
<meta itemprop="property" content="is_mid"/>
</div>

# tf_agents.trajectories.time_step.TimeStep

## Class `TimeStep`

Returned with every call to `step` and `reset` on an environment.





Defined in [`trajectories/time_step.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py).

<!-- Placeholder for "Used in" -->

A `TimeStep` contains the data emitted by an environment at each step of
interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
NumPy array or a dict or list of arrays), and an associated `reward` and
`discount`.

The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
`TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
will equal `StepType.MID.

#### Attributes:

* <b>`step_type`</b>: a `Tensor` or array of `StepType` enum values.
* <b>`reward`</b>: a `Tensor` or array of reward values.
* <b>`discount`</b>: A discount value in the range `[0, 1]`.
* <b>`observation`</b>: A NumPy array, or a nested dict, list or tuple of arrays.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    step_type,
    reward,
    discount,
    observation
)
```

Create new instance of TimeStep(step_type, reward, discount, observation)



## Properties

<h3 id="step_type"><code>step_type</code></h3>



<h3 id="reward"><code>reward</code></h3>



<h3 id="discount"><code>discount</code></h3>



<h3 id="observation"><code>observation</code></h3>





## Methods

<h3 id="is_first"><code>is_first</code></h3>

``` python
is_first()
```



<h3 id="is_last"><code>is_last</code></h3>

``` python
is_last()
```



<h3 id="is_mid"><code>is_mid</code></h3>

``` python
is_mid()
```





