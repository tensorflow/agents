<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.policies.scripted_py_policy.ScriptedPyPolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="info_spec"/>
<meta itemprop="property" content="policy_state_spec"/>
<meta itemprop="property" content="policy_step_spec"/>
<meta itemprop="property" content="time_step_spec"/>
<meta itemprop="property" content="trajectory_spec"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action"/>
<meta itemprop="property" content="get_initial_state"/>
</div>

# tf_agents.policies.scripted_py_policy.ScriptedPyPolicy

## Class `ScriptedPyPolicy`

Returns actions from the given configuration.

Inherits From: [`Base`](../../../tf_agents/policies/py_policy/Base.md)



Defined in [`policies/scripted_py_policy.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/policies/scripted_py_policy.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    time_step_spec,
    action_spec,
    action_script
)
```

Instantiates the scripted policy.

The Action  script can be configured through gin. e.g:

ScriptedPyPolicy.action_script = [
    (1, {  "action1": [[5, 2], [1, 3]],
           "action2": [[4, 6]]},),
    (0, {  "action1": [[8, 1], [9, 2]],
           "action2": [[1, 2]]},),
    (2, {  "action1": [[1, 1], [3, 2]],
           "action2": [[8, 2]]},),
]

In this case the first action is executed once, the second scripted action
is disabled and skipped. Then the third listed action is executed for two
steps.

#### Args:

* <b>`time_step_spec`</b>: A time_step_spec for the policy will interact
    with.
* <b>`action_spec`</b>: An action_spec for the environment the policy will interact
    with.
* <b>`action_script`</b>: A list of 2-tuples of the form (n, nest) where the nest of
    actions follow the action_spec. Each action will be executed for n
    steps.



## Properties

<h3 id="action_spec"><code>action_spec</code></h3>

Describes the ArraySpecs of the np.Array returned by `action()`.

`action` can be a single np.Array, or a nested dict, list or tuple of
np.Array.

#### Returns:

A single BoundedArraySpec, or a nested dict, list or tuple of
`BoundedArraySpec` objects, which describe the shape and
dtype of each np.Array returned by `action()`.

<h3 id="info_spec"><code>info_spec</code></h3>

Describes the Arrays emitted as info by `action()`.

#### Returns:

A nest of ArraySpec which describe the shape and dtype of each Array
emitted as `info` by `action()`.

<h3 id="policy_state_spec"><code>policy_state_spec</code></h3>

Describes the arrays expected by functions with `policy_state` as input.

#### Returns:

A single BoundedArraySpec, or a nested dict, list or tuple of
`BoundedArraySpec` objects, which describe the shape and
dtype of each np.Array returned by `action()`.

<h3 id="policy_step_spec"><code>policy_step_spec</code></h3>

Describes the output of `action()`.

#### Returns:

A nest of ArraySpec which describe the shape and dtype of each Array
emitted by `action()`.

<h3 id="time_step_spec"><code>time_step_spec</code></h3>

Describes the `TimeStep` np.Arrays expected by `action(time_step)`.

#### Returns:

A `TimeStep` namedtuple with `ArraySpec` objects instead of np.Array,
which describe the shape, dtype and name of each array expected by
`action()`.

<h3 id="trajectory_spec"><code>trajectory_spec</code></h3>

Describes the data collected when using this policy with an environment.

#### Returns:

A `Trajectory` containing all array specs associated with the
time_step_spec and policy_step_spec of this policy.



## Methods

<h3 id="action"><code>action</code></h3>

``` python
action(
    time_step,
    policy_state=()
)
```

Generates next action given the time_step and policy_state.


#### Args:

* <b>`time_step`</b>: A `TimeStep` tuple corresponding to `time_step_spec()`.
* <b>`policy_state`</b>: An optional previous policy_state.


#### Returns:

A PolicyStep named tuple containing:
  `action`: A nest of action Arrays matching the `action_spec()`.
  `state`: A nest of policy states to be fed into the next call to action.
  `info`: Optional side information such as action log probabilities.

<h3 id="get_initial_state"><code>get_initial_state</code></h3>

``` python
get_initial_state(batch_size=None)
```

Returns an initial state usable by the policy.

#### Args:

* <b>`batch_size`</b>: An optional batch size.


#### Returns:

An initial policy state.



