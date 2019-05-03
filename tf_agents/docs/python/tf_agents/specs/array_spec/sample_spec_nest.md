<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.array_spec.sample_spec_nest" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.array_spec.sample_spec_nest

Samples the given nest of specs.

``` python
tf_agents.specs.array_spec.sample_spec_nest(
    structure,
    rng,
    outer_dims=()
)
```



Defined in [`specs/array_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/array_spec.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`structure`</b>: An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
* <b>`rng`</b>: A numpy RandomState to use for the sampling.
* <b>`outer_dims`</b>: An optional list/tuple specifying outer dimensions to add to the
    spec shape before sampling.


#### Returns:

A nest of sampled values following the ArraySpec definition.