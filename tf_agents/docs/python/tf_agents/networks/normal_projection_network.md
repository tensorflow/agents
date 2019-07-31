<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.normal_projection_network" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.networks.normal_projection_network

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/networks/normal_projection_network.py">View
source</a>

Project inputs to a normal distribution object.

<!-- Placeholder for "Used in" -->


## Classes

[`class NormalProjectionNetwork`](../../tf_agents/networks/normal_projection_network/NormalProjectionNetwork.md): Generates a tfp.distribution.Normal by predicting a mean and std.

## Functions

[`tanh_squash_to_spec(...)`](../../tf_agents/networks/normal_projection_network/tanh_squash_to_spec.md):
Maps inputs with arbitrary range to range defined by spec using `tanh`.
