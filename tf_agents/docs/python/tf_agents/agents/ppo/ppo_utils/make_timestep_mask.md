<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.ppo.ppo_utils.make_timestep_mask" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.agents.ppo.ppo_utils.make_timestep_mask

Create a mask for final incomplete episodes and episode transitions.

``` python
tf_agents.agents.ppo.ppo_utils.make_timestep_mask(batched_next_time_step)
```



Defined in [`agents/ppo/ppo_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo/ppo_utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`batched_next_time_step`</b>: Next timestep, doubly-batched
    [batch_dim, time_dim, ...].


#### Returns:

A mask, type tf.float32, that is 0.0 for all between-episode timesteps
  (batched_next_time_step is FIRST), or where the episode is
  not compelete, so the return computation would not be correct.