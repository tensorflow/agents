<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.distribution_spec.nested_distributions_from_specs" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.distribution_spec.nested_distributions_from_specs

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/specs/distribution_spec.py">View
source</a>

Builds a nest of distributions from a nest of specs.

``` python
tf_agents.specs.distribution_spec.nested_distributions_from_specs(
    specs,
    parameters
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`specs`</b>: A nest of distribution specs.
* <b>`parameters`</b>: A nest of distribution kwargs.


#### Returns:

Nest of distribution instances with the same structure as the given specs.
