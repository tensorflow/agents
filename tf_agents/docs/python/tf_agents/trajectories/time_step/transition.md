<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step.transition" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.time_step.transition

Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

``` python
tf_agents.trajectories.time_step.transition(
    observation,
    reward,
    discount=1.0
)
```



Defined in [`trajectories/time_step.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py).

<!-- Placeholder for "Used in" -->

For TF transitions, the batch size is inferred from the shape of `reward`.

If `discount` is a scalar, and `observation` contains Tensors,
then `discount` will be broadcasted to match `reward.shape`.

#### Args:

* <b>`observation`</b>: A NumPy array, tensor, or a nested dict, list or tuple of
    arrays or tensors.
* <b>`reward`</b>: A scalar, or 1D NumPy array, or tensor.
* <b>`discount`</b>: (optional) A scalar, or 1D NumPy array, or tensor.


#### Returns:

A `TimeStep`.


#### Raises:

* <b>`ValueError`</b>: If observations are tensors but reward's statically known rank
    is not `0` or `1`.