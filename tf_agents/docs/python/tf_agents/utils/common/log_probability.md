<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.log_probability" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.log_probability

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Computes log probability of actions given distribution.

``` python
tf_agents.utils.common.log_probability(
    distributions,
    actions,
    action_spec
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`distributions`</b>: A possibly batched tuple of distributions.
* <b>`actions`</b>: A possibly batched action tuple.
* <b>`action_spec`</b>: A nested tuple representing the action spec.


#### Returns:

A Tensor representing the log probability of each action in the batch.
