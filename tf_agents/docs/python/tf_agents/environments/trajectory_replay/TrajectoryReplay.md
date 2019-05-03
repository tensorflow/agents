<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.trajectory_replay.TrajectoryReplay" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
</div>

# tf_agents.environments.trajectory_replay.TrajectoryReplay

## Class `TrajectoryReplay`

A helper that replays a policy against given `Trajectory` observations.





Defined in [`environments/trajectory_replay.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/trajectory_replay.py).

<!-- Placeholder for "Used in" -->



<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Creates a TrajectoryReplay object.

TrajectoryReplay.run returns the actions and policy info of the new policy
assuming it saw the observations from the given trajectory.

#### Args:

* <b>`policy`</b>: A tf_policy.Base policy.
* <b>`time_major`</b>: If `True`, the tensors in `trajectory` passed to method `run`
    are assumed to have shape `[time, batch, ...]`.  Otherwise (default)
    they are assumed to have shape `[batch, time, ...]`.


#### Raises:

* <b>`ValueError`</b>:     If policy is not an instance of tf_policy.Base.



## Methods

<h3 id="run"><code>run</code></h3>

``` python
run(
    trajectory,
    policy_state=None
)
```

Apply the policy to trajectory steps and store actions/info.

If `self.time_major == True`, the tensors in `trajectory` are assumed to
have shape `[time, batch, ...]`.  Otherwise they are assumed to
have shape `[batch, time, ...]`.

#### Args:

* <b>`trajectory`</b>: The `Trajectory` to run against.
    If the replay class was created with `time_major=True`, then
    the tensors in trajectory must be shaped `[time, batch, ...]`.
    Otherwise they must be shaped `[batch, time, ...]`.
* <b>`policy_state`</b>: (optional) A nest Tensor with initial step policy state.


#### Returns:

* <b>`output_actions`</b>: A nest of the actions that the policy took.
    If the replay class was created with `time_major=True`, then
    the tensors here will be shaped `[time, batch, ...]`.  Otherwise
    they'll be shaped `[batch, time, ...]`.
* <b>`output_policy_info`</b>: A nest of the policy info that the policy emitted.
    If the replay class was created with `time_major=True`, then
    the tensors here will be shaped `[time, batch, ...]`.  Otherwise
    they'll be shaped `[batch, time, ...]`.
* <b>`policy_state`</b>: A nest Tensor with final step policy state.


#### Raises:

* <b>`TypeError`</b>: If `policy_state` structure doesn't match
    `self.policy.policy_state_spec`, or `trajectory` structure doesn't
    match `self.policy.trajectory_spec`.
* <b>`ValueError`</b>: If `policy_state` doesn't match
    `self.policy.policy_state_spec`, or `trajectory` structure doesn't
    match `self.policy.trajectory_spec`.
* <b>`ValueError`</b>: If `trajectory` lacks two outer dims.



