<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils.stack_nested_arrays" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.nest_utils.stack_nested_arrays

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py">View
source</a>

Stack/batch a list of nested numpy arrays.

``` python
tf_agents.utils.nest_utils.stack_nested_arrays(nested_arrays)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`nested_arrays`</b>: A list of nested numpy arrays of the same shape/structure.


#### Returns:

A nested array containing batched items, where each batched item is obtained by
stacking corresponding items from the list of nested_arrays.
