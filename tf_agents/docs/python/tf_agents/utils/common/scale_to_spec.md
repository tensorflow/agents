<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.scale_to_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.scale_to_spec

Shapes and scales a batch into the given spec bounds.

``` python
tf_agents.utils.common.scale_to_spec(
    tensor,
    spec
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`tensor`</b>: A [batch x n] tensor with values in the range of [-1, 1].
* <b>`spec`</b>: (BoundedTensorSpec) to use for scaling the action.

#### Returns:

A batch scaled the given spec bounds.