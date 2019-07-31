<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.transpose_batch_time" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.transpose_batch_time

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Transposes the batch and time dimensions of a Tensor.

``` python
tf_agents.utils.common.transpose_batch_time(x)
```



<!-- Placeholder for "Used in" -->

If the input tensor has rank < 2 it returns the original tensor. Retains as
much of the static shape information as possible.

#### Args:

* <b>`x`</b>: A Tensor.


#### Returns:

x transposed along the first two dimensions.
