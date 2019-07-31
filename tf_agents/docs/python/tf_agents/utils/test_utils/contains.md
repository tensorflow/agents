<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.test_utils.contains" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.test_utils.contains

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/test_utils.py">View
source</a>

Check if all items in list2 are in list1.

``` python
tf_agents.utils.test_utils.contains(
    list1,
    list2
)
```



<!-- Placeholder for "Used in" -->

This function handles the case when the parameters are lists of np.arrays
(which wouldn't be handled by something like .issubset(...)

#### Args:

* <b>`list1`</b>: List which may or may not contain list2.
* <b>`list2`</b>: List to check if included in list 1.

#### Returns:

A boolean indicating whether list2 is contained in list1.
