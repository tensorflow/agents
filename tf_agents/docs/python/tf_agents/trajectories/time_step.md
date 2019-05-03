<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.trajectories.time_step

TimeStep representing a step in the environment.



Defined in [`trajectories/time_step.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class StepType`](../../tf_agents/trajectories/time_step/StepType.md): Defines the status of a `TimeStep` within a sequence.

[`class TimeStep`](../../tf_agents/trajectories/time_step/TimeStep.md): Returned with every call to `step` and `reset` on an environment.

## Functions

[`restart(...)`](../../tf_agents/trajectories/time_step/restart.md): Returns a `TimeStep` with `step_type` set equal to `StepType.FIRST`.

[`termination(...)`](../../tf_agents/trajectories/time_step/termination.md): Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

[`time_step_spec(...)`](../../tf_agents/trajectories/time_step/time_step_spec.md): Returns a `TimeStep` spec given the observation_spec.

[`transition(...)`](../../tf_agents/trajectories/time_step/transition.md): Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

[`truncation(...)`](../../tf_agents/trajectories/time_step/truncation.md): Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

