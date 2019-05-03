<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.policies.policy_saver.PolicySaver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="save"/>
</div>

# tf_agents.policies.policy_saver.PolicySaver

## Class `PolicySaver`

A `PolicySaver` allows you to save a `tf_policy.Policy` to `SavedModel`.





Defined in [`policies/policy_saver.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/policies/policy_saver.py).

<!-- Placeholder for "Used in" -->

The `save()` method exports a saved model to the requested export location.
The SavedModel that is exported can be loaded via
`tf.compat.v2.saved_model.load` (or `tf.saved_model.load` in TF2).  It
will have available signatures (concrete functions): `action` and
`get_initial_state`.

Usage:
```python

my_policy = agent.collect_policy
saver = PolicySaver(policy, batch_size=None)

for i in range(...):
  agent.train(...)
  if i % 100 == 0:
    saver.save('policy_%d' % global_step)
```

To load and use the saved policy directly:

```python
saved_policy = tf.compat.v2.saved_model.load('policy_0')
policy_state = saved_policy.get_initial_state(batch_size=3)
time_step = ...
while True:
  policy_step = saved_policy.action(time_step, policy_state)
  policy_state = policy_step.state
  time_step = f(policy_step.action)
  ...
```

If using the flattened (signature) version, you will be limited to using
dicts keyed by the specs' name fields.

```python
saved_policy = tf.compat.v2.saved_model.load('policy_0')
get_initial_state_fn = saved_policy.signatures['get_initial_state']
action_fn = saved_policy.signatures['action']

policy_state_dict = get_initial_state_fn(batch_size=3)
time_step_dict = ...
while True:
  time_step_state = dict(time_step_dict)
  time_step_state.update(policy_state_dict)
  policy_step_dict = action_fn(time_step_state)
  policy_state_dict = extract_policy_state_fields(policy_step_dict)
  action_dict = extract_action_fields(policy_step_dict)
  time_step_dict = f(action_dict)
  ...
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    policy,
    batch_size=None,
    use_nest_path_signatures=True,
    seed=None
)
```

Initialize PolicySaver for  TF policy `policy`.

#### Args:

* <b>`policy`</b>: A TF Policy.
* <b>`batch_size`</b>: The number of batch entries the policy will process at a time.
    This must be either `None` (unknown batch size) or a python integer.
* <b>`use_nest_path_signatures`</b>: SavedModel spec signatures will be created based
    on the sructure of the specs. Otherwise all specs must have unique
    names.
* <b>`seed`</b>: Random seed for the `policy.action` call, if any (this should
    usually be `None`, except for testing).


#### Raises:

* <b>`TypeError`</b>: If `policy` is not an instance of TFPolicy.
* <b>`ValueError`</b>: If use_nest_path_signatures is not used and any of the
    following `policy` specs are missing names, or the names collide:
    `policy.time_step_spec`, `policy.action_spec`,
    `policy.policy_state_spec`, `policy.info_spec`.
* <b>`ValueError`</b>: If `batch_size` is not either `None` or a python integer > 0.
* <b>`NotImplementedError`</b>: If created from TF1 with eager mode disabled.



## Methods

<h3 id="save"><code>save</code></h3>

``` python
save(export_dir)
```

Save the policy to the given `export_dir`.



