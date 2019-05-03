<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.SacAgent" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="collect_data_spec"/>
<meta itemprop="property" content="collect_policy"/>
<meta itemprop="property" content="debug_summaries"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="policy"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="summarize_grads_and_vars"/>
<meta itemprop="property" content="time_step_spec"/>
<meta itemprop="property" content="train_sequence_length"/>
<meta itemprop="property" content="train_step_counter"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="actor_loss"/>
<meta itemprop="property" content="alpha_loss"/>
<meta itemprop="property" content="critic_loss"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.agents.SacAgent

## Class `SacAgent`

A SAC Agent.

Inherits From: [`TFAgent`](../../tf_agents/agents/tf_agent/TFAgent.md)

### Aliases:

* Class `tf_agents.agents.SacAgent`
* Class `tf_agents.agents.sac.sac_agent.SacAgent`



Defined in [`agents/sac/sac_agent.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/sac/sac_agent.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Creates a SAC Agent.

#### Args:

* <b>`time_step_spec`</b>: A `TimeStep` spec of the expected time_steps.
* <b>`action_spec`</b>: A nest of BoundedTensorSpec representing the actions.
* <b>`critic_network`</b>: A function critic_network((observations, actions)) that
    returns the q_values for each observation and action.
* <b>`actor_network`</b>: A function actor_network(observation, action_spec) that
    returns action distribution.
* <b>`actor_optimizer`</b>: The optimizer to use for the actor network.
* <b>`critic_optimizer`</b>: The default optimizer to use for the critic network.
* <b>`alpha_optimizer`</b>: The default optimizer to use for the alpha variable.
* <b>`actor_policy_ctor`</b>: The policy class to use.
* <b>`target_update_tau`</b>: Factor for soft update of the target networks.
* <b>`target_update_period`</b>: Period for soft update of the target networks.
* <b>`td_errors_loss_fn`</b>:  A function for computing the elementwise TD errors
    loss.
* <b>`gamma`</b>: A discount factor for future rewards.
* <b>`reward_scale_factor`</b>: Multiplicative scale for the reward.
* <b>`initial_log_alpha`</b>: Initial value for log_alpha.
* <b>`target_entropy`</b>: The target average policy entropy, for updating alpha.
* <b>`gradient_clipping`</b>: Norm length to clip gradients.
* <b>`debug_summaries`</b>: A bool to gather debug summaries.
* <b>`summarize_grads_and_vars`</b>: If True, gradient and network variable summaries
    will be written during training.
* <b>`train_step_counter`</b>: An optional counter to increment every time the train
    op is run.  Defaults to the global_step.
* <b>`name`</b>: The name of this agent. All variables in this module will fall under
    that name. Defaults to the class name.



## Properties

<h3 id="action_spec"><code>action_spec</code></h3>

TensorSpec describing the action produced by the agent.

#### Returns:

An single BoundedTensorSpec, or a nested dict, list or tuple of
`BoundedTensorSpec` objects, which describe the shape and
dtype of each action Tensor.

<h3 id="collect_data_spec"><code>collect_data_spec</code></h3>

Returns a `Trajectory` spec, as expected by the `collect_policy`.

#### Returns:

A `Trajectory` spec.

<h3 id="collect_policy"><code>collect_policy</code></h3>

Return a policy that can be used to collect data from the environment.

#### Returns:

A `tf_policy.Base` object.

<h3 id="debug_summaries"><code>debug_summaries</code></h3>



<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="policy"><code>policy</code></h3>

Return the current policy held by the agent.

#### Returns:

A `tf_policy.Base` object.

<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> assert list(a.submodules) == [b, c]
>>> assert list(b.submodules) == [c]
>>> assert list(c.submodules) == []

#### Returns:

A sequence of all submodules.

<h3 id="summarize_grads_and_vars"><code>summarize_grads_and_vars</code></h3>



<h3 id="time_step_spec"><code>time_step_spec</code></h3>

Describes the `TimeStep` tensors expected by the agent.

#### Returns:

A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
which describe the shape, dtype and name of each tensor.

<h3 id="train_sequence_length"><code>train_sequence_length</code></h3>

The number of time steps needed in experience tensors passed to `train`.

Train requires experience to be a `Trajectory` containing tensors shaped
`[B, T, ...]`.  This argument describes the value of `T` required.

For example, for non-RNN DQN training, `T=2` because DQN requires single
transitions.

If this value is `None`, then `train` can handle an unknown `T` (it can be
determined at runtime from the data).  Most RNN-based agents fall into
this category.

#### Returns:

The number of time steps needed in experience tensors passed to `train`.
May be `None` to mean no constraint.

<h3 id="train_step_counter"><code>train_step_counter</code></h3>



<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).

<h3 id="variables"><code>variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).



## Methods

<h3 id="__delattr__"><code>__delattr__</code></h3>

``` python
__delattr__(name)
```



<h3 id="__setattr__"><code>__setattr__</code></h3>

``` python
__setattr__(
    name,
    value
)
```

Support self.foo = trackable syntax.

<h3 id="actor_loss"><code>actor_loss</code></h3>

``` python
actor_loss(
    time_steps,
    weights=None
)
```

Computes the actor_loss for SAC training.

#### Args:

* <b>`time_steps`</b>: A batch of timesteps.
* <b>`weights`</b>: Optional scalar or elementwise (per-batch-entry) importance
    weights.


#### Returns:

* <b>`actor_loss`</b>: A scalar actor loss.

<h3 id="alpha_loss"><code>alpha_loss</code></h3>

``` python
alpha_loss(
    time_steps,
    weights=None
)
```

Computes the alpha_loss for EC-SAC training.

#### Args:

* <b>`time_steps`</b>: A batch of timesteps.
* <b>`weights`</b>: Optional scalar or elementwise (per-batch-entry) importance
    weights.


#### Returns:

* <b>`alpha_loss`</b>: A scalar alpha loss.

<h3 id="critic_loss"><code>critic_loss</code></h3>

``` python
critic_loss(
    time_steps,
    actions,
    next_time_steps,
    td_errors_loss_fn,
    gamma=1.0,
    reward_scale_factor=1.0,
    weights=None
)
```

Computes the critic loss for SAC training.

#### Args:

* <b>`time_steps`</b>: A batch of timesteps.
* <b>`actions`</b>: A batch of actions.
* <b>`next_time_steps`</b>: A batch of next timesteps.
* <b>`td_errors_loss_fn`</b>: A function(td_targets, predictions) to compute
    elementwise (per-batch-entry) loss.
* <b>`gamma`</b>: Discount for future rewards.
* <b>`reward_scale_factor`</b>: Multiplicative factor to scale rewards.
* <b>`weights`</b>: Optional scalar or elementwise (per-batch-entry) importance
    weights.


#### Returns:

* <b>`critic_loss`</b>: A scalar critic loss.

<h3 id="initialize"><code>initialize</code></h3>

``` python
initialize()
```

Initializes the agent.

#### Returns:

An operation that can be used to initialize the agent.


#### Raises:

* <b>`RuntimeError`</b>: If the class was not initialized properly (`super.__init__`
    was not called).

<h3 id="train"><code>train</code></h3>

``` python
train(
    experience,
    weights=None
)
```

Trains the agent.

#### Args:

* <b>`experience`</b>: A batch of experience data in the form of a `Trajectory`. The
    structure of `experience` must match that of `self.policy.step_spec`.
    All tensors in `experience` must be shaped `[batch, time, ...]` where
    `time` must be equal to `self.required_experience_time_steps` if that
    property is not `None`.
* <b>`weights`</b>: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
    containing weights to be used when calculating the total train loss.
    Weights are typically multiplied elementwise against the per-batch loss,
    but the implementation is up to the Agent.


#### Returns:

A `LossInfo` loss tuple containing loss and info tensors.
- In eager mode, the loss values are first calculated, then a train step
  is performed before they are returned.
- In graph mode, executing any or all of the loss tensors
  will first calculate the loss value(s), then perform a train step,
  and return the pre-train-step `LossInfo`.


#### Raises:

* <b>`TypeError`</b>: If experience is not type `Trajectory`.  Or if experience
    does not match `self.collect_data_spec` structure types.
* <b>`ValueError`</b>: If experience tensors' time axes are not compatible with
    `self.train_sequene_length`.  Or if experience does not match
    `self.collect_data_spec` structure.
* <b>`RuntimeError`</b>: If the class was not initialized properly (`super.__init__`
    was not called).

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
...     return tf.matmul(x, self.w)

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

>>> mod = MyModule()
>>> mod(tf.ones([8, 32]))
<tf.Tensor: ...>
>>> mod.w
<tf.Variable ...'my_module/w:0'>

#### Args:

* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.



