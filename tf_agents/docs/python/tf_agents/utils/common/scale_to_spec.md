<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.scale_to_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.scale_to_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Shapes and scales a batch into the given spec bounds.

``` python
tf_agents.utils.common.scale_to_spec(
    tensor,
    spec
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`tensor`</b>: A [batch x n] tensor with values in the range of [-1, 1].
* <b>`spec`</b>: (BoundedTensorSpec) to use for scaling the action.

#### Returns:

A batch scaled the given spec bounds.
