<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.from_transition" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.trajectory.from_transition

Returns a `Trajectory` given transitions.

``` python
tf_agents.trajectories.trajectory.from_transition(
    time_step,
    action_step,
    next_time_step
)
```



Defined in [`trajectories/trajectory.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py).

<!-- Placeholder for "Used in" -->

`from_transition` is used by a driver to convert sequence of transitions into
a `Trajectory` for efficient storage. Then an agent (e.g.
`ppo_agent.PPOAgent`) converts it back to transitions by invoking
`to_transition`.

#### Args:

* <b>`time_step`</b>: A `time_step.TimeStep` representing the first step in a
    transition.
* <b>`action_step`</b>: A `policy_step.PolicyStep` representing actions corresponding
    to observations from time_step.
* <b>`next_time_step`</b>: A `time_step.TimeStep` representing the second step in a
    transition.