<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.batched_py_environment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.environments.batched_py_environment

Treat multiple non-batch environments as a single batch environment.



Defined in [`environments/batched_py_environment.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/batched_py_environment.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class BatchedPyEnvironment`](../../tf_agents/environments/batched_py_environment/BatchedPyEnvironment.md): Batch together multiple py environments and act as a single batch.

## Functions

[`fast_map_structure(...)`](../../tf_agents/environments/batched_py_environment/fast_map_structure.md): List tf.nest.map_structure, but skipping the slow assert_same_structure.

[`stack_time_steps(...)`](../../tf_agents/environments/batched_py_environment/stack_time_steps.md): Given a list of TimeStep, combine to one with a batch dimension.

[`unstack_actions(...)`](../../tf_agents/environments/batched_py_environment/unstack_actions.md): Returns a list of actions from potentially nested batch of actions.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

