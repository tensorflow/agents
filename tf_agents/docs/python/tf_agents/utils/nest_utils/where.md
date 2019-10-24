<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.where" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.where

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py">View
source</a>

Generalization of tf.compat.v1.where supporting nests as the outputs.

```python
tf_agents.utils.nest_utils.where(
    condition,
    true_outputs,
    false_outputs
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`condition`</b>: A boolean Tensor of shape [B,].
*   <b>`true_outputs`</b>: Tensor or nested tuple of Tensors of any dtype, each
    with shape [B, ...], to be split based on `condition`.
*   <b>`false_outputs`</b>: Tensor or nested tuple of Tensors of any dtype, each
    with shape [B, ...], to be split based on `condition`.

#### Returns:

Interleaved output from `true_outputs` and `false_outputs` based on `condition`.
