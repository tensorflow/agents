<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.tensor_spec.to_nest_placeholder" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.tensor_spec.to_nest_placeholder

Converts a nest of TensorSpecs to a nest of matching placeholders.

``` python
tf_agents.specs.tensor_spec.to_nest_placeholder(
    nested_tensor_specs,
    default=None,
    name_scope='',
    outer_dims=()
)
```



Defined in [`specs/tensor_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/tensor_spec.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`nested_tensor_specs`</b>: A nest of tensor specs.
* <b>`default`</b>: Optional constant value to set as a default for the placeholder.
* <b>`name_scope`</b>: String name for the scope to create the placeholders in.
* <b>`outer_dims`</b>: Optional leading dimensions for the placeholder.


#### Returns:

A nest of placeholders matching the given tensor spec.


#### Raises:

* <b>`ValueError`</b>: If a default is provided outside of the allowed types, or if
    default is a np.array that does not match the spec shape.