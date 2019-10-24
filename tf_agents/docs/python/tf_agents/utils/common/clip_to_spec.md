<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.clip_to_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.clip_to_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Clips value to a given bounded tensor spec.

``` python
tf_agents.utils.common.clip_to_spec(
    value,
    spec
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`value`</b>: (tensor) value to be clipped.
* <b>`spec`</b>: (BoundedTensorSpec) spec containing min. and max. values for clipping.

#### Returns:

* <b>`clipped_value`</b>: (tensor) `value` clipped to be compatible with `spec`.