<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.entropy" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.entropy

Computes total entropy of distribution.

``` python
tf_agents.utils.common.entropy(
    distributions,
    action_spec
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`distributions`</b>: A possibly batched tuple of distributions.
* <b>`action_spec`</b>: A nested tuple representing the action spec.


#### Returns:

A Tensor representing the entropy of each distribution in the batch.
Assumes actions are independent, so that marginal entropies of each action
may be summed.