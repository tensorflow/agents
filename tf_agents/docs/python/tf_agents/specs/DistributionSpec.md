<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.DistributionSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="builder"/>
<meta itemprop="property" content="distribution_parameters"/>
<meta itemprop="property" content="input_params_spec"/>
<meta itemprop="property" content="sample_spec"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build_distribution"/>
</div>

# tf_agents.specs.DistributionSpec

## Class `DistributionSpec`

Describes a tfp.distribution.Distribution.



### Aliases:

* Class `tf_agents.specs.DistributionSpec`
* Class `tf_agents.specs.distribution_spec.DistributionSpec`



Defined in [`specs/distribution_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/distribution_spec.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    builder,
    input_params_spec,
    sample_spec,
    **distribution_parameters
)
```

Creates a DistributionSpec.

#### Args:

* <b>`builder`</b>: Callable function(**params) which returns a Distribution
    following the spec.
* <b>`input_params_spec`</b>: Nest of tensor_specs describing the tensor parameters
    required for building the described distribution.
* <b>`sample_spec`</b>: Data type of the output samples of the described
    distribution.
* <b>`**distribution_parameters`</b>: Extra parameters for building the distribution.



## Properties

<h3 id="builder"><code>builder</code></h3>

Returns the `distribution_builder` of the spec.

<h3 id="distribution_parameters"><code>distribution_parameters</code></h3>

Returns the `distribution_parameters` of the spec.

<h3 id="input_params_spec"><code>input_params_spec</code></h3>

Returns the `input_params_spec` of the spec.

<h3 id="sample_spec"><code>sample_spec</code></h3>

Returns the `sample_spec` of the spec.



## Methods

<h3 id="build_distribution"><code>build_distribution</code></h3>

``` python
build_distribution(**distribution_parameters)
```

Creates an instance of the described distribution.

The spec's paramers are updated with the given ones.
#### Args:

* <b>`**distribution_parameters`</b>: Kwargs update the spec's distribution
    parameters.


#### Returns:

Distribution instance.



