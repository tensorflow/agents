<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.get_outer_rank" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.get_outer_rank

Compares tensors to specs to determine the number of batch dimensions.

``` python
tf_agents.utils.nest_utils.get_outer_rank(
    tensors,
    specs
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

For each tensor, it checks the dimensions with respect to specs and
returns the number of batch dimensions if all nested tensors and
specs agree with each other.

#### Args:

* <b>`tensors`</b>: Nested list/tuple/dict of Tensors.
* <b>`specs`</b>: Nested list/tuple/dict of TensorSpecs, describing the shape of
    unbatched tensors.


#### Returns:

The number of outer dimensions for all Tensors (zero if all are
  unbatched or empty).

#### Raises:

* <b>`ValueError`</b>: If
    1. Any of the tensors or specs have shapes with ndims == None, or
    2. The shape of Tensors are not compatible with specs, or
    3. A mix of batched and unbatched tensors are provided.
    4. The tensors are batched but have an incorrect number of outer dims.