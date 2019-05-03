<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.to_transition" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.trajectory.to_transition

Create a transition from a trajectory or two adjacent trajectories.

``` python
tf_agents.trajectories.trajectory.to_transition(
    trajectory,
    next_trajectory=None
)
```



Defined in [`trajectories/trajectory.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py).

<!-- Placeholder for "Used in" -->

**NOTE** If `next_trajectory` is not provided, tensors of `trajectory` are
sliced along their *second* (`time`) dimension; for example:

```
time_steps.step_type = trajectory.step_type[:,:-1]
time_steps.observation = trajectory.observation[:,:-1]
next_time_steps.observation = trajectory.observation[:,1:]
next_time_steps. step_type = trajectory. next_step_type[:,:-1]
next_time_steps.reward = trajectory.reward[:,:-1]
next_time_steps. discount = trajectory. discount[:,:-1]

```
Notice that reward and discount for time_steps are undefined, therefore filled
with zero.

#### Args:

* <b>`trajectory`</b>: An instance of `Trajectory`. The tensors in Trajectory must have
    shape `[ B, T, ...]` when next_trajectory is None.
* <b>`next_trajectory`</b>: (optional) An instance of `Trajectory`.


#### Returns:

A tuple `(time_steps, policy_steps, next_time_steps)`.  The `reward` and
`discount` fields of `time_steps` are filled with zeros because these
cannot be deduced (please do not use them).