<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.batch_nested_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.batch_nested_tensors

Add batch dimension if needed to nested tensors while checking their specs.

``` python
tf_agents.utils.nest_utils.batch_nested_tensors(
    tensors,
    specs=None
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

If specs is None, a batch dimension is added to each tensor.
If specs are provided, each tensor is compared to the corresponding spec,
and a batch dimension is added only if the tensor doesn't already have it.

For each tensor, it checks the dimensions with respect to specs, and adds an
extra batch dimension if it doesn't already have it.

#### Args:

* <b>`tensors`</b>: Nested list/tuple or dict of Tensors.
* <b>`specs`</b>: Nested list/tuple or dict of TensorSpecs, describing the shape of the
    non-batched Tensors.


#### Returns:

A nested batched version of each tensor.

#### Raises:

* <b>`ValueError`</b>: if the tensors and specs have incompatible dimensions or shapes.