<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.drivers.dynamic_step_driver.DynamicStepDriver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="env"/>
<meta itemprop="property" content="observers"/>
<meta itemprop="property" content="policy"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
</div>

# tf_agents.drivers.dynamic_step_driver.DynamicStepDriver

## Class `DynamicStepDriver`

A driver that takes N steps in an environment using a tf.while_loop.

Inherits From: [`Driver`](../../../tf_agents/drivers/driver/Driver.md)



Defined in [`drivers/dynamic_step_driver.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/drivers/dynamic_step_driver.py).

<!-- Placeholder for "Used in" -->

The while loop will run num_steps in the environment, only counting steps that
result in an environment transition, i.e. (time_step, action, next_time_step).
If a step results in environment resetting, i.e. time_step.is_last() and
next_time_step.is_first() (traj.is_boundary()), this is not counted toward the
num_steps.

As environments run batched time_steps, the counters for all batch elements
are summed, and execution stops when the total exceeds num_steps. When
batch_size > 1, there is no guarantee that exactly num_steps are taken -- it
may be more but never less.

This termination condition can be overridden in subclasses by implementing the
self._loop_condition_fn() method.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Creates a DynamicStepDriver.

#### Args:

* <b>`env`</b>: A tf_environment.Base environment.
* <b>`policy`</b>: A tf_policy.Base policy.
* <b>`observers`</b>: A list of observers that are updated after every step in
    the environment. Each observer is a callable(time_step.Trajectory).
* <b>`num_steps`</b>: The number of steps to take in the environment.


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
    maximum_iterations=None
)
```

Takes steps in the environment using the policy while updating observers.

#### Args:

* <b>`time_step`</b>: optional initial time_step. If None, it will use the
    current_time_step of the environment. Elements should be shape
    [batch_size, ...].
* <b>`policy_state`</b>: optional initial state for the policy.
* <b>`maximum_iterations`</b>: Optional maximum number of iterations of the while
    loop to run. If provided, the cond output is AND-ed with an additional
    condition ensuring the number of iterations executed is no greater than
    maximum_iterations.


#### Returns:

* <b>`time_step`</b>: TimeStep named tuple with final observation, reward, etc.
* <b>`policy_state`</b>: Tensor with final step policy state.



