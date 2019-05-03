<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.tensor_spec.sample_bounded_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.tensor_spec.sample_bounded_spec

Samples uniformily the given bounded spec.

``` python
tf_agents.specs.tensor_spec.sample_bounded_spec(
    spec,
    seed=None,
    outer_dims=None
)
```



Defined in [`specs/tensor_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/tensor_spec.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`spec`</b>: A BoundedSpec to sample.
* <b>`seed`</b>: A seed used for sampling ops
* <b>`outer_dims`</b>: An optional `Tensor` specifying outer dimensions to add to
    the spec shape before sampling.

#### Returns:

An Tensor sample of the requested spec.