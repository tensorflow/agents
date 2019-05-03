<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.PPOAgent" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="actor_net"/>
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
<meta itemprop="property" content="adaptive_kl_loss"/>
<meta itemprop="property" content="compute_advantages"/>
<meta itemprop="property" content="compute_return_and_advantage"/>
<meta itemprop="property" content="entropy_regularization_loss"/>
<meta itemprop="property" content="get_epoch_loss"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="kl_cutoff_loss"/>
<meta itemprop="property" content="kl_penalty_loss"/>
<meta itemprop="property" content="l2_regularization_loss"/>
<meta itemprop="property" content="policy_gradient_loss"/>
<meta itemprop="property" content="train"/>
<meta itemprop="property" content="update_adaptive_kl_beta"/>
<meta itemprop="property" content="value_estimation_loss"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.agents.PPOAgent

## Class `PPOAgent`

A PPO Agent.

Inherits From: [`TFAgent`](../../tf_agents/agents/tf_agent/TFAgent.md)

### Aliases:

* Class `tf_agents.agents.PPOAgent`
* Class `tf_agents.agents.ppo.ppo_agent.PPOAgent`



Defined in [`agents/ppo/ppo_agent.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo/ppo_agent.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Creates a PPO Agent.

#### Args:

* <b>`time_step_spec`</b>: A `TimeStep` spec of the expected time_steps.
* <b>`action_spec`</b>: A nest of BoundedTensorSpec representing the actions.
* <b>`optimizer`</b>: Optimizer to use for the agent.
* <b>`actor_net`</b>: A function actor_net(observations, action_spec) that returns
    tensor of action distribution params for each observation. Takes nested
    observation and returns nested action.
* <b>`value_net`</b>: A function value_net(time_steps) that returns value tensor from
    neural net predictions for each observation. Takes nested observation
    and returns batch of value_preds.
* <b>`importance_ratio_clipping`</b>: Epsilon in clipped, surrogate PPO objective.
    For more detail, see explanation at the top of the doc.
* <b>`lambda_value`</b>: Lambda parameter for TD-lambda computation.
* <b>`discount_factor`</b>: Discount factor for return computation.
* <b>`entropy_regularization`</b>: Coefficient for entropy regularization loss term.
* <b>`policy_l2_reg`</b>: Coefficient for l2 regularization of policy weights.
* <b>`value_function_l2_reg`</b>: Coefficient for l2 regularization of value function
    weights.
* <b>`value_pred_loss_coef`</b>: Multiplier for value prediction loss to balance with
    policy gradient loss.
* <b>`num_epochs`</b>: Number of epochs for computing policy updates.
* <b>`use_gae`</b>: If True (default False), uses generalized advantage estimation
    for computing per-timestep advantage. Else, just subtracts value
    predictions from empirical return.
* <b>`use_td_lambda_return`</b>: If True (default False), uses td_lambda_return for
    training value function. (td_lambda_return = gae_advantage +
    value_predictions)
* <b>`normalize_rewards`</b>: If true, keeps moving variance of rewards and
    normalizes incoming rewards.
* <b>`reward_norm_clipping`</b>: Value above an below to clip normalized reward.
* <b>`normalize_observations`</b>: If true, keeps moving mean and variance of
    observations and normalizes incoming observations.
* <b>`log_prob_clipping`</b>: +/- value for clipping log probs to prevent inf / NaN
    values.  Default: no clipping.
* <b>`kl_cutoff_factor`</b>: If policy KL changes more than this much for any single
    timestep, adds a squared KL penalty to loss function.
* <b>`kl_cutoff_coef`</b>: Loss coefficient for kl cutoff term.
* <b>`initial_adaptive_kl_beta`</b>: Initial value for beta coefficient of adaptive
    kl penalty.
* <b>`adaptive_kl_target`</b>: Desired kl target for policy updates. If actual kl is
    far from this target, adaptive_kl_beta will be updated.
* <b>`adaptive_kl_tolerance`</b>: A tolerance for adaptive_kl_beta. Mean KL above (1
    + tol) * adaptive_kl_target, or below (1 - tol) * adaptive_kl_target,
    will cause adaptive_kl_beta to be updated.
* <b>`gradient_clipping`</b>: Norm length to clip gradients.  Default: no clipping.
* <b>`check_numerics`</b>: If true, adds tf.debugging.check_numerics to help find
    NaN / Inf values. For debugging only.
* <b>`debug_summaries`</b>: A bool to gather debug summaries.
* <b>`summarize_grads_and_vars`</b>: If true, gradient summaries will be written.
* <b>`train_step_counter`</b>: An optional counter to increment every time the train
    op is run.  Defaults to the global_step.
* <b>`name`</b>: The name of this agent. All variables in this module will fall
    under that name. Defaults to the class name.


#### Raises:

* <b>`ValueError`</b>: If the actor_net is not a DistributionNetwork.



## Properties

<h3 id="action_spec"><code>action_spec</code></h3>

TensorSpec describing the action produced by the agent.

#### Returns:

An single BoundedTensorSpec, or a nested dict, list or tuple of
`BoundedTensorSpec` objects, which describe the shape and
dtype of each action Tensor.

<h3 id="actor_net"><code>actor_net</code></h3>

Returns actor_net TensorFlow template function.

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

<h3 id="adaptive_kl_loss"><code>adaptive_kl_loss</code></h3>

``` python
adaptive_kl_loss(
    kl_divergence,
    debug_summaries=False
)
```



<h3 id="compute_advantages"><code>compute_advantages</code></h3>

``` python
compute_advantages(
    rewards,
    returns,
    discounts,
    value_preds
)
```

Compute advantages, optionally using GAE.

Based on baselines ppo1 implementation. Removes final timestep, as it needs
to use this timestep for next-step value prediction for TD error
computation.

#### Args:

* <b>`rewards`</b>: Tensor of per-timestep rewards.
* <b>`returns`</b>: Tensor of per-timestep returns.
* <b>`discounts`</b>: Tensor of per-timestep discounts. Zero for terminal timesteps.
* <b>`value_preds`</b>: Cached value estimates from the data-collection policy.


#### Returns:

* <b>`advantages`</b>: Tensor of length (len(rewards) - 1), because the final
    timestep is just used for next-step value prediction.

<h3 id="compute_return_and_advantage"><code>compute_return_and_advantage</code></h3>

``` python
compute_return_and_advantage(
    next_time_steps,
    value_preds
)
```

Compute the Monte Carlo return and advantage.

Normalazation will be applied to the computed returns and advantages if
it's enabled.

#### Args:

* <b>`next_time_steps`</b>: batched tensor of TimeStep tuples after action is taken.
* <b>`value_preds`</b>: Batched value predction tensor. Should have one more entry in
    time index than time_steps, with the final value corresponding to the
    value prediction of the final state.


#### Returns:

tuple of (return, normalized_advantage), both are batched tensors.

<h3 id="entropy_regularization_loss"><code>entropy_regularization_loss</code></h3>

``` python
entropy_regularization_loss(
    time_steps,
    current_policy_distribution,
    weights,
    debug_summaries=False
)
```

Create regularization loss tensor based on agent parameters.

<h3 id="get_epoch_loss"><code>get_epoch_loss</code></h3>

``` python
get_epoch_loss(
    time_steps,
    actions,
    act_log_probs,
    returns,
    normalized_advantages,
    action_distribution_parameters,
    weights,
    train_step,
    debug_summaries
)
```

Compute the loss and create optimization op for one training epoch.

All tensors should have a single batch dimension.

#### Args:

* <b>`time_steps`</b>: A minibatch of TimeStep tuples.
* <b>`actions`</b>: A minibatch of actions.
* <b>`act_log_probs`</b>: A minibatch of action probabilities (probability under the
    sampling policy).
* <b>`returns`</b>: A minibatch of per-timestep returns.
* <b>`normalized_advantages`</b>: A minibatch of normalized per-timestep advantages.
* <b>`action_distribution_parameters`</b>: Parameters of data-collecting action
    distribution. Needed for KL computation.
* <b>`weights`</b>: Optional scalar or element-wise (per-batch-entry) importance
    weights.  Includes a mask for invalid timesteps.
* <b>`train_step`</b>: A train_step variable to increment for each train step.
    Typically the global_step.
* <b>`debug_summaries`</b>: True if debug summaries should be created.


#### Returns:

A tf_agent.LossInfo named tuple with the total_loss and all intermediate
  losses in the extra field contained in a PPOLossInfo named tuple.

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

<h3 id="kl_cutoff_loss"><code>kl_cutoff_loss</code></h3>

``` python
kl_cutoff_loss(
    kl_divergence,
    debug_summaries=False
)
```



<h3 id="kl_penalty_loss"><code>kl_penalty_loss</code></h3>

``` python
kl_penalty_loss(
    time_steps,
    action_distribution_parameters,
    current_policy_distribution,
    weights,
    debug_summaries=False
)
```

Compute a loss that penalizes policy steps with high KL.

Based on KL divergence from old (data-collection) policy to new (updated)
policy.

All tensors should have a single batch dimension.

#### Args:

* <b>`time_steps`</b>: TimeStep tuples with observations for each timestep. Used for
    computing new action distributions.
* <b>`action_distribution_parameters`</b>: Action distribution params of the data
    collection policy, used for reconstruction old action distributions.
* <b>`current_policy_distribution`</b>: The policy distribution, evaluated on all
    time_steps.
* <b>`weights`</b>: Optional scalar or element-wise (per-batch-entry) importance
    weights.  Inlcudes a mask for invalid timesteps.
* <b>`debug_summaries`</b>: True if debug summaries should be created.


#### Returns:

* <b>`kl_penalty_loss`</b>: The sum of a squared penalty for KL over a constant
    threshold, plus an adaptive penalty that encourages updates toward a
    target KL divergence.

<h3 id="l2_regularization_loss"><code>l2_regularization_loss</code></h3>

``` python
l2_regularization_loss(debug_summaries=False)
```



<h3 id="policy_gradient_loss"><code>policy_gradient_loss</code></h3>

``` python
policy_gradient_loss(
    time_steps,
    actions,
    sample_action_log_probs,
    advantages,
    current_policy_distribution,
    weights,
    debug_summaries=False
)
```

Create tensor for policy gradient loss.

All tensors should have a single batch dimension.

#### Args:

* <b>`time_steps`</b>: TimeSteps with observations for each timestep.
* <b>`actions`</b>: Tensor of actions for timesteps, aligned on index.
* <b>`sample_action_log_probs`</b>: Tensor of sample probability of each action.
* <b>`advantages`</b>: Tensor of advantage estimate for each timestep, aligned on
    index. Works better when advantage estimates are normalized.
* <b>`current_policy_distribution`</b>: The policy distribution, evaluated on all
    time_steps.
* <b>`weights`</b>: Optional scalar or element-wise (per-batch-entry) importance
    weights.  Includes a mask for invalid timesteps.
* <b>`debug_summaries`</b>: True if debug summaries should be created.


#### Returns:

* <b>`policy_gradient_loss`</b>: A tensor that will contain policy gradient loss for
    the on-policy experience.

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

<h3 id="update_adaptive_kl_beta"><code>update_adaptive_kl_beta</code></h3>

``` python
update_adaptive_kl_beta(kl_divergence)
```

Create update op for adaptive KL penalty coefficient.

#### Args:

* <b>`kl_divergence`</b>: KL divergence of old policy to new policy for all
    timesteps.


#### Returns:

* <b>`update_op`</b>: An op which runs the update for the adaptive kl penalty term.

<h3 id="value_estimation_loss"><code>value_estimation_loss</code></h3>

``` python
value_estimation_loss(
    time_steps,
    returns,
    weights,
    debug_summaries=False
)
```

Computes the value estimation loss for actor-critic training.

All tensors should have a single batch dimension.

#### Args:

* <b>`time_steps`</b>: A batch of timesteps.
* <b>`returns`</b>: Per-timestep returns for value function to predict. (Should come
    from TD-lambda computation.)
* <b>`weights`</b>: Optional scalar or element-wise (per-batch-entry) importance
    weights.  Includes a mask for invalid timesteps.
* <b>`debug_summaries`</b>: True if debug summaries should be created.


#### Returns:

* <b>`value_estimation_loss`</b>: A scalar value_estimation_loss loss.

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



