<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.tensor_spec.to_placeholder_with_default" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.tensor_spec.to_placeholder_with_default

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/specs/tensor_spec.py">View
source</a>

Creates a placeholder from TensorSpec.

``` python
tf_agents.specs.tensor_spec.to_placeholder_with_default(
    default,
    spec,
    outer_dims=()
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`default`</b>: A constant value of output type dtype.
* <b>`spec`</b>: Instance of TensorSpec
* <b>`outer_dims`</b>: Optional leading dimensions of the placeholder.


#### Returns:

An instance of tf.placeholder.
