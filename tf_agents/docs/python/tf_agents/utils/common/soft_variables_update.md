<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.soft_variables_update" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.soft_variables_update

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Performs a soft/hard update of variables from the source to the target.

``` python
tf_agents.utils.common.soft_variables_update(
    source_variables,
    target_variables,
    tau=1.0,
    sort_variables_by_name=False
)
```



<!-- Placeholder for "Used in" -->

For each variable v_t in target variables and its corresponding variable v_s
in source variables, a soft update is:
v_t = (1 - tau) * v_t + tau * v_s

When tau is 1.0 (the default), then it does a hard update:
v_t = v_s

#### Args:

*   <b>`source_variables`</b>: list of source variables.
*   <b>`target_variables`</b>: list of target variables.
*   <b>`tau`</b>: A float scalar in [0, 1]. When tau is 1.0 (the default), we do
    a hard update.
*   <b>`sort_variables_by_name`</b>: A bool, when True would sort the variables
    by name before doing the update.

#### Returns:

An operation that updates target variables from source variables.

#### Raises:

* <b>`ValueError`</b>: if tau is not in [0, 1].