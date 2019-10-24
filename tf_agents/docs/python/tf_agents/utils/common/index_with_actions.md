<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.index_with_actions" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.index_with_actions

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Index into q_values using actions.

``` python
tf_agents.utils.common.index_with_actions(
    q_values,
    actions,
    multi_dim_actions=False
)
```



<!-- Placeholder for "Used in" -->

Note: this supports multiple outer dimensions (e.g. time, batch etc).

#### Args:

*   <b>`q_values`</b>: A float tensor of shape [outer_dim1, ... outer_dimK,
    action_dim1, ..., action_dimJ].
*   <b>`actions`</b>: An int tensor of shape [outer_dim1, ... outer_dimK] if
    multi_dim_actions=False [outer_dim1, ... outer_dimK, J] if
    multi_dim_actions=True I.e. in the multidimensional case,
    actions[outer_dim1, ... outer_dimK] is a vector [actions_1, ..., actions_J]
    where each element actions_j is an action in the range [0, num_actions_j).
    While in the single dimensional case, actions[outer_dim1, ... outer_dimK] is
    a scalar.
*   <b>`multi_dim_actions`</b>: whether the actions are multidimensional.

#### Returns:

A [outer_dim1, ... outer_dimK] tensor of q_values for the given actions.

#### Raises:

* <b>`ValueError`</b>: If actions have unknown rank.