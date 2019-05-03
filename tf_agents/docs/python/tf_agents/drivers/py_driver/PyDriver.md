<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.drivers.py_driver.PyDriver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="env"/>
<meta itemprop="property" content="observers"/>
<meta itemprop="property" content="policy"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
</div>

# tf_agents.drivers.py_driver.PyDriver

## Class `PyDriver`

A driver that runs a python policy in a python environment.

Inherits From: [`Driver`](../../../tf_agents/drivers/driver/Driver.md)



Defined in [`drivers/py_driver.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/drivers/py_driver.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    env,
    policy,
    observers,
    max_steps=None,
    max_episodes=None
)
```

A driver that runs a python policy in a python environment.

#### Args:

* <b>`env`</b>: A py_environment.Base environment.
* <b>`policy`</b>: A py_policy.Base policy.
* <b>`observers`</b>: A list of observers that are notified after every step
    in the environment. Each observer is a callable(trajectory.Trajectory).
* <b>`max_steps`</b>: Optional maximum number of steps for each run() call.
    Also see below.  Default: 0.
* <b>`max_episodes`</b>: Optional maximum number of episodes for each run() call.
    At least one of max_steps or max_episodes must be provided. If both
    are set, run() terminates when at least one of the conditions is
    satisfied.  Default: 0.


#### Raises:

* <b>`ValueError`</b>: If both max_steps and max_episodes are None.



## Properties

<h3 id="env"><code>env</code></h3>



<h3 id="observers"><code>observers</code></h3>



<h3 id="policy"><code>policy</code></h3>





## Methods

<h3 id="run"><code>run</code></h3>

``` python
run(
    time_step,
    policy_state=()
)
```

Run policy in environment given initial time_step and policy_state.

#### Args:

* <b>`time_step`</b>: The initial time_step.
* <b>`policy_state`</b>: The initial policy_state.


#### Returns:

A tuple (final time_step, final policy_state).



