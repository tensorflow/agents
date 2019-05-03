<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.tf_environment" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.environments.tf_environment

TensorFlow RL Environment API.



Defined in [`environments/tf_environment.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py).

<!-- Placeholder for "Used in" -->

Represents a task to be solved, an environment has to define three methods:
`reset`, `current_time_step` and `step`.

- The reset() method returns current time_step after resetting the environment.
- The current_time_step() method returns current time_step initializing the
environmet if needed. Only needed in graph mode.
- The step(action) method applies the action and returns the new time_step.

## Classes

[`class TFEnvironment`](../../tf_agents/environments/tf_environment/TFEnvironment.md): Abstract base class for TF RL environments.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

