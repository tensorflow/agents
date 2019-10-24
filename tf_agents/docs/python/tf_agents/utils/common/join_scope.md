<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.join_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.join_scope

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Joins a parent and child scope using `/`, checking for empty/none.

``` python
tf_agents.utils.common.join_scope(
    parent_scope,
    child_scope
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`parent_scope`</b>: (string) parent/prefix scope.
* <b>`child_scope`</b>: (string) child/suffix scope.

#### Returns:

joined scope: (string) parent and child scopes joined by /.
