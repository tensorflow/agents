<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.last" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.trajectories.trajectory.last

Create a Trajectory transitioning between StepTypes `MID` and `LAST`.

``` python
tf_agents.trajectories.trajectory.last(
    observation,
    action,
    policy_info,
    reward,
    discount
)
```



Defined in [`trajectories/trajectory.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py).

<!-- Placeholder for "Used in" -->

All inputs may be batched.

The input `discount` is used to infer the outer shape of the inputs,
as it is always expected to be a singleton array with scalar inner shape.

#### Args:

* <b>`observation`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`;
    all shaped `[T, ...]`.
* <b>`action`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`;
    all shaped `[T, ...]`.
* <b>`policy_info`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`;
    all shaped `[T, ...]`.
* <b>`reward`</b>: (possibly nested tuple of) `Tensor` or `np.ndarray`;
    all shaped `[T, ...]`.
* <b>`discount`</b>: A floating point vector `Tensor` or `np.ndarray`;
    shaped `[T]` (optional).


#### Returns:

A `Trajectory` instance.