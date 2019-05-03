<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step.termination" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.time_step.termination

Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

``` python
tf_agents.trajectories.time_step.termination(
    observation,
    reward
)
```



Defined in [`trajectories/time_step.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`observation`</b>: A NumPy array, tensor, or a nested dict, list or tuple of
    arrays or tensors.
* <b>`reward`</b>: A scalar, or 1D NumPy array, or tensor.


#### Returns:

A `TimeStep`.


#### Raises:

* <b>`ValueError`</b>: If observations are tensors but reward's statically known rank
    is not `0` or `1`.