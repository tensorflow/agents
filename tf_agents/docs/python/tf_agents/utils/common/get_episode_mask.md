<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.get_episode_mask" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.get_episode_mask

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Create a mask that is 0.0 for all final steps, 1.0 elsewhere.

``` python
tf_agents.utils.common.get_episode_mask(time_steps)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`time_steps`</b>: A TimeStep namedtuple representing a batch of steps.


#### Returns:

A float32 Tensor with 0s where step_type == LAST and 1s otherwise.
