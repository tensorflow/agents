<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step.restart" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.time_step.restart

Returns a `TimeStep` with `step_type` set equal to `StepType.FIRST`.

``` python
tf_agents.trajectories.time_step.restart(
    observation,
    batch_size=None
)
```



Defined in [`trajectories/time_step.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`observation`</b>: A NumPy array, tensor, or a nested dict, list or tuple of
    arrays or tensors.
* <b>`batch_size`</b>: (Optional) A python or tensorflow integer scalar.


#### Returns:

A `TimeStep`.