<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step.truncation" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.time_step.truncation

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py">View
source</a>

Returns a `TimeStep` with `step_type` set to
<a href="../../../tf_agents/trajectories/time_step/StepType.md#LAST"><code>StepType.LAST</code></a>.

``` python
tf_agents.trajectories.time_step.truncation(
    observation,
    reward,
    discount=1.0
)
```



<!-- Placeholder for "Used in" -->

If `discount` is a scalar, and `observation` contains Tensors,
then `discount` will be broadcasted to match `reward.shape`.

#### Args:

*   <b>`observation`</b>: A NumPy array, tensor, or a nested dict, list or tuple
    of arrays or tensors.
*   <b>`reward`</b>: A scalar, or 1D NumPy array, or tensor.
*   <b>`discount`</b>: (optional) A scalar, or 1D NumPy array, or tensor.

#### Returns:

A `TimeStep`.

#### Raises:

*   <b>`ValueError`</b>: If observations are tensors but reward's statically
    known rank is not `0` or `1`.
