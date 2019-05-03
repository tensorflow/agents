<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.flatten_multi_batched_nested_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.flatten_multi_batched_nested_tensors

Reshape tensors to contain only one batch dimension.

``` python
tf_agents.utils.nest_utils.flatten_multi_batched_nested_tensors(
    tensors,
    specs
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

For each tensor, it checks the number of extra dimensions beyond those in
the spec, and reshapes tensor to have only one batch dimension.
NOTE: Each tensor's batch dimensions must be the same.

#### Args:

* <b>`tensors`</b>: Nested list/tuple or dict of batched Tensors.
* <b>`specs`</b>: Nested list/tuple or dict of TensorSpecs, describing the shape of the
    non-batched Tensors.


#### Returns:

A nested version of each tensor with a single batch dimension.
A list of the batch dimensions which were flattened.

#### Raises:

* <b>`ValueError`</b>: if the tensors and specs have incompatible dimensions or shapes.