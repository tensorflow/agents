<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.to_transition_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.trajectory.to_transition_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py">View
source</a>

Create a transition spec from a trajectory spec.

```python
tf_agents.trajectories.trajectory.to_transition_spec(trajectory_spec)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`trajectory_spec`</b>: An instance of `Trajectory` representing
    trajectory specs.

#### Returns:

A tuple `(time_steps, policy_steps, next_time_steps)` specs.
