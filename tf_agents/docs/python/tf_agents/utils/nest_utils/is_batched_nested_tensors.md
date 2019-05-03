<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.is_batched_nested_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.is_batched_nested_tensors

Compares tensors to specs to determine if all tensors are batched or not.

``` python
tf_agents.utils.nest_utils.is_batched_nested_tensors(
    tensors,
    specs,
    num_outer_dims=1
)
```



Defined in [`utils/nest_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py).

<!-- Placeholder for "Used in" -->

For each tensor, it checks the dimensions with respect to specs and returns
True if all tensors are batched and False if all tensors are unbatched, and
raises a ValueError if the shapes are incompatible or a mix of batched and
unbatched tensors are provided.

#### Args:

* <b>`tensors`</b>: Nested list/tuple/dict of Tensors.
* <b>`specs`</b>: Nested list/tuple/dict of Tensors describing the shape of unbatched
    tensors.
* <b>`num_outer_dims`</b>: The integer number of dimensions that are considered batch
    dimensions.  Default 1.


#### Returns:

True if all Tensors are batched and False if all Tensors are unbatched.

#### Raises:

* <b>`ValueError`</b>: If
    1. Any of the tensors or specs have shapes with ndims == None, or
    2. The shape of Tensors are not compatible with specs, or
    3. A mix of batched and unbatched tensors are provided.
    4. The tensors are batched but have an incorrect number of outer dims.