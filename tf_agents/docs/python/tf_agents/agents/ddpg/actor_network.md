<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.ddpg.actor_network" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.agents.ddpg.actor_network

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ddpg/actor_network.py">View
source</a>

Sample Actor network to use with DDPG agents.

<!-- Placeholder for "Used in" -->

Note: This network scales actions to fit the given spec by using `tanh`. Due to
the nature of the `tanh` function, actions near the spec bounds cannot be
returned.

## Classes

[`class ActorNetwork`](../../../tf_agents/agents/ddpg/actor_network/ActorNetwork.md): Creates an actor network.

