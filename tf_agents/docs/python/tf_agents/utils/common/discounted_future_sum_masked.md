<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.discounted_future_sum_masked" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.discounted_future_sum_masked

Discounted future sum of batch-major values.

``` python
tf_agents.utils.common.discounted_future_sum_masked(
    values,
    gamma,
    num_steps,
    episode_lengths
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`values`</b>: A Tensor of shape [batch_size, total_steps] and dtype float32.
* <b>`gamma`</b>: A float discount value.
* <b>`num_steps`</b>: A positive integer number of future steps to sum.
* <b>`episode_lengths`</b>: A vector shape [batch_size] with num_steps per episode.


#### Returns:

A Tensor of shape [batch_size, total_steps], where each entry is the
  discounted sum as in discounted_future_sum, except with values after
  the end of episode_lengths masked to 0.


#### Raises:

* <b>`ValueError`</b>: If values is not of rank 2, or if total_steps is not defined.