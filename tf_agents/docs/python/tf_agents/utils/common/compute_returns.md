<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.compute_returns" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.compute_returns

Compute the return from each index in an episode.

``` python
tf_agents.utils.common.compute_returns(
    rewards,
    discounts
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`rewards`</b>: Tensor of per-timestep reward in the episode.
* <b>`discounts`</b>: Tensor of per-timestep discount factor. Should be 0 for final
    step of each episode.


#### Returns:

Tensor of per-timestep cumulative returns.