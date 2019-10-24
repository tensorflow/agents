<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.td3.td3_agent" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.agents.td3.td3_agent

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/agents/td3/td3_agent.py">View
source</a>

Twin Delayed Deep Deterministic policy gradient (TD3) agent.

<!-- Placeholder for "Used in" -->

TD3 extends DDPG by adding an extra critic network and using the minimum of the
two critic values to reduce overestimation bias.

"Addressing Function Approximation Error in Actor-Critic Methods"
by Fujimoto et al.

For the full paper, see https://arxiv.org/abs/1802.09477.

## Classes

[`class Td3Agent`](../../../tf_agents/agents/Td3Agent.md): A TD3 Agent.

[`class Td3Info`](../../../tf_agents/agents/td3/td3_agent/Td3Info.md)

