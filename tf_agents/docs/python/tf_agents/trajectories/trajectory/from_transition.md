<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.from_transition" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.trajectory.from_transition

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py">View
source</a>

Returns a `Trajectory` given transitions.

``` python
tf_agents.trajectories.trajectory.from_transition(
    time_step,
    action_step,
    next_time_step
)
```



<!-- Placeholder for "Used in" -->

`from_transition` is used by a driver to convert sequence of transitions into a
`Trajectory` for efficient storage. Then an agent (e.g.
<a href="../../../tf_agents/agents/PPOAgent.md"><code>ppo_agent.PPOAgent</code></a>)
converts it back to transitions by invoking `to_transition`.

Note that this method does not add a time dimension to the Tensors in the
resulting `Trajectory`. This means that if your transitions don't already
include a time dimension, the `Trajectory` cannot be passed to `agent.train()`.

#### Args:

*   <b>`time_step`</b>: A
    <a href="../../../tf_agents/trajectories/time_step/TimeStep.md"><code>time_step.TimeStep</code></a>
    representing the first step in a transition.
*   <b>`action_step`</b>: A
    <a href="../../../tf_agents/trajectories/policy_step/PolicyStep.md"><code>policy_step.PolicyStep</code></a>
    representing actions corresponding to observations from time_step.
*   <b>`next_time_step`</b>: A
    <a href="../../../tf_agents/trajectories/time_step/TimeStep.md"><code>time_step.TimeStep</code></a>
    representing the second step in a transition.
