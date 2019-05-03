<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.value_ops.generalized_advantage_estimation" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.value_ops.generalized_advantage_estimation

Computes generalized advantage estimation (GAE).

``` python
tf_agents.utils.value_ops.generalized_advantage_estimation(
    values,
    final_value,
    discounts,
    rewards,
    td_lambda=1.0,
    time_major=True
)
```



Defined in [`utils/value_ops.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/value_ops.py).

<!-- Placeholder for "Used in" -->

For theory, see
"High-Dimensional Continuous Control Using Generalized Advantage Estimation"
by John Schulman, Philipp Moritz et al.
See https://arxiv.org/abs/1506.02438 for full paper.

Define abbreviations:
  (B) batch size representing number of trajectories
  (T) number of steps per trajectory

#### Args:

* <b>`values`</b>: Tensor with shape [T, B] representing value estimates.
* <b>`final_value`</b>: Tensor with shape [B] representing value estimate at t=T.
* <b>`discounts`</b>: Tensor with shape [T, B] representing discounts received by
    following the behavior policy.
* <b>`rewards`</b>: Tensor with shape [T, B] representing rewards received by following
    the behavior policy.
* <b>`td_lambda`</b>: A float32 scalar between [0, 1]. It's used for variance reduction
    in temporal difference.
* <b>`time_major`</b>: A boolean indicating whether input tensors are time major.
    False means input tensors have shape [B, T].


#### Returns:

A tensor with shape [T, B] representing advantages. Shape is [B, T] when
time_major is false.