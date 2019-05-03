<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.unbatch_nested_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.unbatch_nested_tensors

Remove the batch dimension if needed from nested tensors using their specs.

``` python
tf_agents.utils.nest_utils.unbatch_nested_tensors(
    tensors,
    specs=None
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

If specs is None, the first dimension of each tensor will be removed.
If specs are provided, each tensor is compared to the corresponding spec,
and the first dimension is removed only if the tensor was batched.

#### Args:

* <b>`tensors`</b>: Nested list/tuple or dict of batched Tensors.
* <b>`specs`</b>: Nested list/tuple or dict of TensorSpecs, describing the shape of the
    non-batched Tensors.


#### Returns:

A nested non-batched version of each tensor.

#### Raises:

* <b>`ValueError`</b>: if the tensors and specs have incompatible dimensions or shapes.