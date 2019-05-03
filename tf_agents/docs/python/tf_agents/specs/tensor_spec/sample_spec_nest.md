<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.tensor_spec.sample_spec_nest" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.tensor_spec.sample_spec_nest

Samples the given nest of specs.

``` python
tf_agents.specs.tensor_spec.sample_spec_nest(
    structure,
    seed=None,
    outer_dims=()
)
```



Defined in [`specs/tensor_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/tensor_spec.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`structure`</b>: An `TensorSpec`, or a nested dict, list or tuple of
      `TensorSpec`s.
* <b>`seed`</b>: A seed used for sampling ops
* <b>`outer_dims`</b>: An optional `Tensor` specifying outer dimensions to add to
    the spec shape before sampling.

#### Returns:

A nest of sampled values following the ArraySpec definition.