<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.drivers.driver.Driver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="env"/>
<meta itemprop="property" content="observers"/>
<meta itemprop="property" content="policy"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
</div>

# tf_agents.drivers.driver.Driver

## Class `Driver`

A driver that takes steps in an environment using a policy.





Defined in [`drivers/driver.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/drivers/driver.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    env,
    policy,
    observers=None
)
```

Creates a Driver.

#### Args:

* <b>`env`</b>: An environment.Base environment.
* <b>`policy`</b>: A policy.Base policy.
* <b>`observers`</b>: A list of observers that are updated after the driver is run.
    Each observer is a callable(Trajectory) that returns the input.
    Trajectory.time_step is a stacked batch [N+1, batch_size, ...] of
    timesteps and Trajectory.action is a stacked batch
    [N, batch_size, ...] of actions in time major form.



## Properties

<h3 id="env"><code>env</code></h3>



<h3 id="observers"><code>observers</code></h3>



<h3 id="policy"><code>policy</code></h3>





## Methods

<h3 id="run"><code>run</code></h3>

``` python
run()
```

Takes steps in the environment and updates observers.



