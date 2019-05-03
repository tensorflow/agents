<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.replicate" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.replicate

Replicates a tensor so as to match the given outer shape.

``` python
tf_agents.utils.common.replicate(
    tensor,
    outer_shape
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

Example:
- t = [[1, 2, 3], [4, 5, 6]] (shape = [2, 3])
- outer_shape = [2, 1]
The shape of the resulting tensor is: [2, 1, 2, 3]
and its content is: [[t], [t]]

#### Args:

* <b>`tensor`</b>: A tf.Tensor.
* <b>`outer_shape`</b>: Outer shape given as a 1D tensor of type
    list, numpy or tf.Tensor.


#### Returns:

The replicated tensor.


#### Raises:

* <b>`ValueError`</b>: when the outer shape is incorrect.