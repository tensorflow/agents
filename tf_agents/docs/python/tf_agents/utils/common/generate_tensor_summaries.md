<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.generate_tensor_summaries" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.generate_tensor_summaries

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Generates various summaries of `tensor` such as histogram, max, min, etc.

``` python
tf_agents.utils.common.generate_tensor_summaries(
    tag,
    tensor,
    step
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`tag`</b>: A namescope tag for the summaries.
* <b>`tensor`</b>: The tensor to generate summaries of.
* <b>`step`</b>: Variable to use for summaries.