<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory.Trajectory" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="step_type"/>
<meta itemprop="property" content="observation"/>
<meta itemprop="property" content="action"/>
<meta itemprop="property" content="policy_info"/>
<meta itemprop="property" content="next_step_type"/>
<meta itemprop="property" content="reward"/>
<meta itemprop="property" content="discount"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="is_boundary"/>
<meta itemprop="property" content="is_first"/>
<meta itemprop="property" content="is_last"/>
<meta itemprop="property" content="is_mid"/>
<meta itemprop="property" content="replace"/>
</div>

# tf_agents.trajectories.trajectory.Trajectory

## Class `Trajectory`

A tuple that represents a trajectory.





Defined in [`trajectories/trajectory.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py).

<!-- Placeholder for "Used in" -->

A `Trajectory` is a sequence of aligned time steps. It captures the
observation, step_type from current time step with the computed action
and policy_info. Discount, reward and next_step_type come from the next
time step.

#### Attributes:

* <b>`step_type`</b>: A `StepType`.
* <b>`observation`</b>: An array (tensor), or a nested dict, list or tuple of arrays
    (tensors) that represents the observation.
* <b>`action`</b>: An array/a tensor, or a nested dict, list or tuple of actions. This
    represents action generated according to the observation.
* <b>`policy_info`</b>: A namedtuple that contains auxiliary information related to the
    action. Note that this does not include the policy/RNN state which was
    used to generate the action.
* <b>`next_step_type`</b>: The `StepType` of the next time step.
* <b>`reward`</b>: A scalar representing the reward of performing the action in an
    environment.
* <b>`discount`</b>: A scalar that representing the discount factor to multiply with
    future rewards.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    step_type,
    observation,
    action,
    policy_info,
    next_step_type,
    reward,
    discount
)
```

Create new instance of Trajectory(step_type, observation, action, policy_info, next_step_type, reward, discount)



## Properties

<h3 id="step_type"><code>step_type</code></h3>



<h3 id="observation"><code>observation</code></h3>



<h3 id="action"><code>action</code></h3>



<h3 id="policy_info"><code>policy_info</code></h3>



<h3 id="next_step_type"><code>next_step_type</code></h3>



<h3 id="reward"><code>reward</code></h3>



<h3 id="discount"><code>discount</code></h3>





## Methods

<h3 id="is_boundary"><code>is_boundary</code></h3>

``` python
is_boundary()
```



<h3 id="is_first"><code>is_first</code></h3>

``` python
is_first()
```



<h3 id="is_last"><code>is_last</code></h3>

``` python
is_last()
```



<h3 id="is_mid"><code>is_mid</code></h3>

``` python
is_mid()
```



<h3 id="replace"><code>replace</code></h3>

``` python
replace(**kwargs)
```

Exposes as namedtuple._replace.

Usage:
```
  new_trajectory = trajectory.replace(policy_info=())
```

This returns a new trajectory with an empty policy_info.

#### Args:

* <b>`**kwargs`</b>: key/value pairs of fields in the trajectory.


#### Returns:

A new `Trajectory`.



