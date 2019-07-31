<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.ppo.ppo_utils.get_distribution_params" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.agents.ppo.ppo_utils.get_distribution_params

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo/ppo_utils.py">View
source</a>

Get the params for an optionally nested action distribution.

``` python
tf_agents.agents.ppo.ppo_utils.get_distribution_params(nested_distribution)
```



<!-- Placeholder for "Used in" -->

Only returns parameters that have tf.Tensor values.

#### Args:

*   <b>`nested_distribution`</b>: The nest of distributions whose parameter
    tensors to extract.

#### Returns:

A nest of distribution parameters. Each leaf is a dict corresponding to one
distribution, with keys as parameter name and values as tensors containing
parameter values.
