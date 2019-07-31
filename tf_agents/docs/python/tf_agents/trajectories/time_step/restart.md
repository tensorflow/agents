<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step.restart" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.time_step.restart

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py">View
source</a>

Returns a `TimeStep` with `step_type` set equal to
<a href="../../../tf_agents/trajectories/time_step/StepType.md#FIRST"><code>StepType.FIRST</code></a>.

``` python
tf_agents.trajectories.time_step.restart(
    observation,
    batch_size=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`observation`</b>: A NumPy array, tensor, or a nested dict, list or tuple
    of arrays or tensors.
*   <b>`batch_size`</b>: (Optional) A python or tensorflow integer scalar.

#### Returns:

A `TimeStep`.
