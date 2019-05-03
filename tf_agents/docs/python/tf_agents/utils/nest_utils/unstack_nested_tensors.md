<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.unstack_nested_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.unstack_nested_tensors

Make list of unstacked nested tensors.

``` python
tf_agents.utils.nest_utils.unstack_nested_tensors(
    tensors,
    specs
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`tensors`</b>: Nested tensors whose first dimension is to be unstacked.
* <b>`specs`</b>: Tensor specs for tensors.


#### Returns:

A list of the unstacked nested tensors.

#### Raises:

* <b>`ValueError`</b>: if the tensors and specs have incompatible dimensions or shapes.