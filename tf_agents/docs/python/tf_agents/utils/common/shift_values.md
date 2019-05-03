<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.shift_values" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.shift_values

Shifts batch-major values in time by some amount.

``` python
tf_agents.utils.common.shift_values(
    values,
    gamma,
    num_steps,
    final_values=None
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`values`</b>: A Tensor of shape [batch_size, total_steps] and dtype float32.
* <b>`gamma`</b>: A float discount value.
* <b>`num_steps`</b>: A nonnegative integer amount to shift values by.
* <b>`final_values`</b>: A float32 Tensor of shape [batch_size] corresponding to the
    values at step num_steps + 1.  Defaults to None (all zeros).


#### Returns:

A Tensor of shape [batch_size, total_steps], where each entry (i, j) is
gamma^num_steps * values[i, j + num_steps] if j + num_steps < total_steps;
gamma^(total_steps - j) * final_values[i] otherwise.


#### Raises:

* <b>`ValueError`</b>: If values is not of rank 2.