<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.discounted_future_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.discounted_future_sum

Discounted future sum of batch-major values.

``` python
tf_agents.utils.common.discounted_future_sum(
    values,
    gamma,
    num_steps
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`values`</b>: A Tensor of shape [batch_size, total_steps] and dtype float32.
* <b>`gamma`</b>: A float discount value.
* <b>`num_steps`</b>: A positive integer number of future steps to sum.


#### Returns:

A Tensor of shape [batch_size, total_steps], where each entry `(i, j)` is
  the result of summing the entries of values starting from
  `gamma^0 * values[i, j]` to
  `gamma^(num_steps - 1) * values[i, j + num_steps - 1]`,
  with zeros padded to values.

  For example, values=[5, 6, 7], gamma=0.9, will result in sequence:
  ```python
  [(5 * 0.9^0 + 6 * 0.9^1 + 7 * 0.9^2), (6 * 0.9^0 + 7 * 0.9^1), 7 * 0.9^0]
  ```


#### Raises:

* <b>`ValueError`</b>: If values is not of rank 2.