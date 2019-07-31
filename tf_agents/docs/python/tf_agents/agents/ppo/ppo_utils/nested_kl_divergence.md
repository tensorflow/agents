<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.ppo.ppo_utils.nested_kl_divergence" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.agents.ppo.ppo_utils.nested_kl_divergence

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo/ppo_utils.py">View
source</a>

Given two nested distributions, sum the KL divergences of the leaves.

``` python
tf_agents.agents.ppo.ppo_utils.nested_kl_divergence(
    nested_from_distribution,
    nested_to_distribution,
    outer_dims=()
)
```



<!-- Placeholder for "Used in" -->
