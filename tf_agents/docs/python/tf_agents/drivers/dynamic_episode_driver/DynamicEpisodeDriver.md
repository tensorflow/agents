<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="env"/>
<meta itemprop="property" content="observers"/>
<meta itemprop="property" content="policy"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
</div>

# tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver

## Class `DynamicEpisodeDriver`

A driver that takes N episodes in an environment using a tf.while_loop.

Inherits From: [`Driver`](../../../tf_agents/drivers/driver/Driver.md)



Defined in [`drivers/dynamic_episode_driver.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/drivers/dynamic_episode_driver.py).

<!-- Placeholder for "Used in" -->

The while loop will run num_episodes in the environment, counting transitions
that result in ending an episode.

As environments run batched time_episodes, the counters for all batch elements
are summed, and execution stops when the total exceeds num_episodes.

This termination condition can be overridden in subclasses by implementing the
self._loop_condition_fn() method.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Creates a DynamicEpisodeDriver.

#### Args:

* <b>`env`</b>: A tf_environment.Base environment.
* <b>`policy`</b>: A tf_policy.Base policy.
* <b>`observers`</b>: A list of observers that are updated after every step in
    the environment. Each observer is a callable(Trajectory).
* <b>`num_episodes`</b>: The number of episodes to take in the environment.


#### Raises:

* <b>`ValueError`</b>:     If env is not a tf_environment.Base or policy is not an instance of
    tf_policy.Base.



## Properties

<h3 id="env"><code>env</code></h3>



<h3 id="observers"><code>observers</code></h3>



<h3 id="policy"><code>policy</code></h3>





## Methods

<h3 id="run"><code>run</code></h3>

``` python
run(
    time_step=None,
    policy_state=None,
    num_episodes=None,
    maximum_iterations=None
)
```

Takes episodes in the environment using the policy and update observers.

If `time_step` and `policy_state` are not provided, `run` will reset the
environment and request an initial state from the policy.

#### Args:

* <b>`time_step`</b>: optional initial time_step. If None, it will be obtained by
    resetting the environment. Elements should be shape [batch_size, ...].
* <b>`policy_state`</b>: optional initial state for the policy. If None, it will be
    obtained from the policy.get_initial_state().
* <b>`num_episodes`</b>: Optional number of episodes to take in the environment. If
    None it would use initial num_episodes.
* <b>`maximum_iterations`</b>: Optional maximum number of iterations of the while
    loop to run. If provided, the cond output is AND-ed with an additional
    condition ensuring the number of iterations executed is no greater than
    maximum_iterations.


#### Returns:

* <b>`time_step`</b>: TimeStep named tuple with final observation, reward, etc.
* <b>`policy_state`</b>: Tensor with final step policy state.



