<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.ppo.ppo_agent" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.agents.ppo.ppo_agent

A PPO Agent.



Defined in [`agents/ppo/ppo_agent.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo/ppo_agent.py).

<!-- Placeholder for "Used in" -->

Implements the PPO algorithm from (Schulman, 2017):
https://arxiv.org/abs/1707.06347

PPO is a simplification of the TRPO algorithm, both of which add stability to
policy gradient RL, while allowing multiple updates per batch of on-policy data,
by limiting the KL divergence between the policy that sampled the data and the
updated policy.

TRPO enforces a hard optimization constraint, but is a complex algorithm, which
often makes it harder to use in practice. PPO approximates the effect of TRPO
by using a soft constraint. There are two methods presented in the paper for
implementing the soft constraint: an adaptive KL loss penalty, and
limiting the objective value based on a clipped version of the policy importance
ratio. This code implements both, and allows the user to use either method or
both by modifying hyperparameters.

The importance ratio clipping is described in eq (7) and the adaptive KL penatly
is described in eq (8) of https://arxiv.org/pdf/1707.06347.pdf
- To disable IR clipping, set the importance_ratio_clipping parameter to 0.0
- To disable the adaptive KL penalty, set the initial_adaptive_kl_beta parameter
  to 0.0
- To disable the fixed KL cutoff penalty, set the kl_cutoff_factor parameter
  to 0.0

In order to compute KL divergence, the replay buffer must store action
distribution parameters from data collection. For now, it is assumed that
continuous actions are represented by a Normal distribution with mean & stddev,
and discrete actions are represented by a Categorical distribution with logits.

Note that the objective function chooses the lower value of the clipped and
unclipped objectives. Thus, if the importance ratio exceeds the clipped bounds,
then the optimizer will still not be incentivized to pass the bounds, as it is
only optimizing the minimum.

Advantage is computed using Generalized Advantage Estimation (GAE):
https://arxiv.org/abs/1506.02438

## Classes

[`class PPOAgent`](../../../tf_agents/agents/PPOAgent.md): A PPO Agent.

[`class PPOLossInfo`](../../../tf_agents/agents/ppo/ppo_agent/PPOLossInfo.md): PPOLossInfo(policy_gradient_loss, value_estimation_loss, l2_regularization_loss, entropy_regularization_loss, kl_penalty_loss)

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

