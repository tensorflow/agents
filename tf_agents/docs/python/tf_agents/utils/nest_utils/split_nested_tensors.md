<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.split_nested_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.split_nested_tensors

Split batched nested tensors, on batch dim (outer dim), into a list.

``` python
tf_agents.utils.nest_utils.split_nested_tensors(
    tensors,
    specs,
    num_or_size_splits
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`tensors`</b>: Nested list/tuple or dict of batched Tensors.
* <b>`specs`</b>: Nested list/tuple or dict of TensorSpecs, describing the shape of the
    non-batched Tensors.
* <b>`num_or_size_splits`</b>: Same as argument for tf.split. Either a 0-D integer
    Tensor indicating the number of splits along batch_dim or a 1-D integer
    Tensor containing the sizes of each output tensor along batch_dim. If a
    scalar then it must evenly divide value.shape[axis]; otherwise the sum of
    sizes along the split dimension must match that of the value.


#### Returns:

A list of nested non-batched version of each tensor, where each list item
  corresponds to one batch item.

#### Raises:

* <b>`ValueError`</b>: if the tensors and specs have incompatible dimensions or shapes.