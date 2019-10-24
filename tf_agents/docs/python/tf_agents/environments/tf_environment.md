<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.tf_environment" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.environments.tf_environment

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/tf_environment.py">View
source</a>

TensorFlow RL Environment API.

<!-- Placeholder for "Used in" -->

Represents a task to be solved, an environment has to define three methods:
`reset`, `current_time_step` and `step`.

- The reset() method returns current time_step after resetting the environment.
- The current_time_step() method returns current time_step initializing the
environmet if needed. Only needed in graph mode.
- The step(action) method applies the action and returns the new time_step.

## Classes

[`class TFEnvironment`](../../tf_agents/environments/tf_environment/TFEnvironment.md): Abstract base class for TF RL environments.

