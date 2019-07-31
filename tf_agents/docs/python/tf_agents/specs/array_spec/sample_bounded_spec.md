<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.array_spec.sample_bounded_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.array_spec.sample_bounded_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/specs/array_spec.py">View
source</a>

Samples the given bounded spec.

``` python
tf_agents.specs.array_spec.sample_bounded_spec(
    spec,
    rng
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`spec`</b>: A BoundedSpec to sample.
* <b>`rng`</b>: A numpy RandomState to use for the sampling.


#### Returns:

An np.array sample of the requested space.
