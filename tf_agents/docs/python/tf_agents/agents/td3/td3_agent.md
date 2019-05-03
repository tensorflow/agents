<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.td3.td3_agent" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.agents.td3.td3_agent

Twin Delayed Deep Deterministic policy gradient (TD3) agent.



Defined in [`agents/td3/td3_agent.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/td3/td3_agent.py).

<!-- Placeholder for "Used in" -->

TD3 extends DDPG by adding an extra critic network and using the minimum of the
two critic values to reduce overestimation bias.

"Addressing Function Approximation Error in Actor-Critic Methods"
by Fujimoto et al.

For the full paper, see https://arxiv.org/abs/1802.09477.

## Classes

[`class Td3Agent`](../../../tf_agents/agents/Td3Agent.md): A TD3 Agent.

[`class Td3Info`](../../../tf_agents/agents/td3/td3_agent/Td3Info.md)

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

