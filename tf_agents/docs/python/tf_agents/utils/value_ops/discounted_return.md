<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.value_ops.discounted_return" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.value_ops.discounted_return

Computes discounted return.

``` python
tf_agents.utils.value_ops.discounted_return(
    rewards,
    discounts,
    final_value=None,
    time_major=True
)
```



Defined in [`utils/value_ops.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/value_ops.py).

<!-- Placeholder for "Used in" -->

```
Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'} + gamma^(T-t+1)*final_value.
```

For details, see
"Reinforcement Learning: An Introduction" Second Edition
by Richard S. Sutton and Andrew G. Barto

Define abbreviations:
(B) batch size representing number of trajectories
(T) number of steps per trajectory

#### Args:

* <b>`rewards`</b>: Tensor with shape [T, B] (or [T]) representing rewards.
* <b>`discounts`</b>: Tensor with shape [T, B] (or [T]) representing discounts.
* <b>`final_value`</b>: Tensor with shape [B] (or [1]) representing value estimate at
    t=T. This is optional, when set, it allows final value to bootstrap the
    reward to go computation. Otherwise it's zero.
* <b>`time_major`</b>: A boolean indicating whether input tensors are time major. False
    means input tensors have shape [B, T].


#### Returns:

A tensor with shape [T, B] (or [T]) representing the discounted returns.
Shape is [B, T] when time_major is false.