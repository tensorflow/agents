<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.ddpg.actor_rnn_network" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.agents.ddpg.actor_rnn_network

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ddpg/actor_rnn_network.py">View
source</a>

Sample recurrent Actor network to use with DDPG agents.

<!-- Placeholder for "Used in" -->

Note: This network scales actions to fit the given spec by using `tanh`. Due to
the nature of the `tanh` function, actions near the spec bounds cannot be
returned.

## Classes

[`class ActorRnnNetwork`](../../../tf_agents/agents/ddpg/actor_rnn_network/ActorRnnNetwork.md): Creates a recurrent actor network.

