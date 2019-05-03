<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.from_episode" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.trajectory.from_episode

Create a Trajectory from tensors representing a single episode.

``` python
tf_agents.trajectories.trajectory.from_episode(
    observation,
    action,
    policy_info,
    reward,
    discount=None
)
```



Defined in [`trajectories/trajectory.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py).

<!-- Placeholder for "Used in" -->

If none of the inputs are tensors, then numpy arrays are generated instead.

If `discount` is not provided, the first entry in `reward` is used to estimate
`T`:

```
reward_0 = tf.nest.flatten(reward)[0]
T = shape(reward_0)[0]
```

In this case, a `discount` of all ones having dtype `float32` is generated.

Notice: all tensors/numpy arrays passed to this function has the same time
dimension T. When the generated trajectory passes through `to_transition`, it
will only return (time_steps, next_time_steps) pair with T-1 in time
dimension, which means the reward at step T is dropped. So if the reward at
step T is important, please make sure the episode passed to this function
contains an additional step.

#### Args:

* <b>`observation`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
    `[T, ...]`.
* <b>`action`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped `[T,
    ...]`.
* <b>`policy_info`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
    `[T, ...]`.
* <b>`reward`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped `[T,
    ...]`.
* <b>`discount`</b>: A floating point vector `Tensor` or `np.ndarray`; shaped `[T]`
    (optional).


#### Returns:

An instance of `Trajectory`.