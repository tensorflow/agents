<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.unstack_nested_arrays" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.unstack_nested_arrays

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py">View
source</a>

Unstack/unbatch a nest of numpy arrays.

``` python
tf_agents.utils.nest_utils.unstack_nested_arrays(nested_array)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`nested_array`</b>: Nest of numpy arrays where each array has shape
    [batch_size, ...].

#### Returns:

A list of length batch_size where each item in the list is a nest having the
same structure as `nested_array`.
