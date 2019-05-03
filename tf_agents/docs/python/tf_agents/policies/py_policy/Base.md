<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.policies.py_policy.Base" />
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

# tf_agents.policies.py_policy.Base

## Class `Base`

Abstract base class for Python Policies.





Defined in [`policies/py_policy.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/policies/py_policy.py).

<!-- Placeholder for "Used in" -->

The `action(time_step, policy_state)` method returns a PolicyStep named tuple
containing the following nested arrays:
  `action`: The action to be applied on the environment.
  `state`: The state of the policy (E.g. RNN state) to be fed into the next
    call to action.
  `info`: Optional side information such as action log probabilities.

For stateful policies, e.g. those containing RNNs, an initial policy state can
be obtained through a call to `get_initial_state()`.

Example of simple use in Python:

  py_env = PyEnvironment()
  policy = PyPolicy()

  time_step = py_env.reset()
  policy_state = policy.get_initial_state()

  acc_reward = 0
  while not time_step.is_last():
    action_step = policy.action(time_step, policy_state)
    policy_state = action_step.state
    time_step = py_env.step(action_step.action)
    acc_reward += time_step.reward

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    time_step_spec,
    action_spec,
    policy_state_spec=(),
    info_spec=()
)
```

Initialization of Base class.

#### Args:

* <b>`time_step_spec`</b>: A `TimeStep` ArraySpec of the expected time_steps.
    Usually provided by the user to the subclass.
* <b>`action_spec`</b>: A nest of BoundedArraySpec representing the actions.
    Usually provided by the user to the subclass.
* <b>`policy_state_spec`</b>: A nest of ArraySpec representing the policy state.
    Provided by the subclass, not directly by the user.
* <b>`info_spec`</b>: A nest of ArraySpec representing the policy info.
    Provided by the subclass, not directly by the user.



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



