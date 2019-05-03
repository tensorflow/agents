<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.policies.py_tf_policy.PyTFPolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="action_spec"/>
<meta itemprop="property" content="info_spec"/>
<meta itemprop="property" content="policy_state_spec"/>
<meta itemprop="property" content="policy_step_spec"/>
<meta itemprop="property" content="session"/>
<meta itemprop="property" content="time_step_spec"/>
<meta itemprop="property" content="trajectory_spec"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="action"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="restore"/>
<meta itemprop="property" content="save"/>
</div>

# tf_agents.policies.py_tf_policy.PyTFPolicy

## Class `PyTFPolicy`

Exposes a Python policy as wrapper over a TF Policy.

Inherits From: [`Base`](../../../tf_agents/policies/py_policy/Base.md), [`SessionUser`](../../../tf_agents/utils/session_utils/SessionUser.md)



Defined in [`policies/py_tf_policy.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/policies/py_tf_policy.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    policy,
    batch_size=None,
    seed=None
)
```

Initializes a new `PyTFPolicy`.

#### Args:

* <b>`policy`</b>: A TF Policy implementing `tf_policy.Base`.
* <b>`batch_size`</b>: (deprecated)
* <b>`seed`</b>: Seed to use if policy performs random actions (optional).



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

<h3 id="session"><code>session</code></h3>

Returns the TensorFlow session-like object used by this object.

#### Returns:

The internal TensorFlow session-like object. If it is `None`, it will
return the current TensorFlow session context manager.


#### Raises:

* <b>`AttributeError`</b>: When no session-like object has been set, and no
    session context manager has been entered.

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

<h3 id="initialize"><code>initialize</code></h3>

``` python
initialize(
    batch_size,
    graph=None
)
```



<h3 id="restore"><code>restore</code></h3>

``` python
restore(
    policy_dir,
    graph=None,
    assert_consumed=True
)
```

Restores the policy from the checkpoint.

#### Args:

* <b>`policy_dir`</b>: Directory with the checkpoint.
* <b>`graph`</b>: A graph, inside which policy the is restored (optional).
* <b>`assert_consumed`</b>: If true, contents of the checkpoint will be checked
    for a match against graph variables.


#### Returns:

* <b>`step`</b>: Global step associated with the restored policy checkpoint.


#### Raises:

* <b>`RuntimeError`</b>: if the policy is not initialized.
* <b>`AssertionError`</b>: if the checkpoint contains variables which do not have
    matching names in the graph, and assert_consumed is set to True.

<h3 id="save"><code>save</code></h3>

``` python
save(
    policy_dir=None,
    graph=None
)
```





