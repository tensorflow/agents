<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.log_probability" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.log_probability

Computes log probability of actions given distribution.

``` python
tf_agents.utils.common.log_probability(
    distributions,
    actions,
    action_spec
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`distributions`</b>: A possibly batched tuple of distributions.
* <b>`actions`</b>: A possibly batched action tuple.
* <b>`action_spec`</b>: A nested tuple representing the action spec.


#### Returns:

A Tensor representing the log probability of each action in the batch.