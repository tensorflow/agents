<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.time_step" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.trajectories.time_step

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/time_step.py">View
source</a>

TimeStep representing a step in the environment.

<!-- Placeholder for "Used in" -->


## Classes

[`class StepType`](../../tf_agents/trajectories/time_step/StepType.md): Defines the status of a `TimeStep` within a sequence.

[`class TimeStep`](../../tf_agents/trajectories/time_step/TimeStep.md): Returned with every call to `step` and `reset` on an environment.

## Functions

[`restart(...)`](../../tf_agents/trajectories/time_step/restart.md): Returns a
`TimeStep` with `step_type` set equal to
<a href="../../tf_agents/trajectories/time_step/StepType.md#FIRST"><code>StepType.FIRST</code></a>.

[`termination(...)`](../../tf_agents/trajectories/time_step/termination.md):
Returns a `TimeStep` with `step_type` set to
<a href="../../tf_agents/trajectories/time_step/StepType.md#LAST"><code>StepType.LAST</code></a>.

[`time_step_spec(...)`](../../tf_agents/trajectories/time_step/time_step_spec.md): Returns a `TimeStep` spec given the observation_spec.

[`transition(...)`](../../tf_agents/trajectories/time_step/transition.md):
Returns a `TimeStep` with `step_type` set equal to
<a href="../../tf_agents/trajectories/time_step/StepType.md#MID"><code>StepType.MID</code></a>.

[`truncation(...)`](../../tf_agents/trajectories/time_step/truncation.md):
Returns a `TimeStep` with `step_type` set to
<a href="../../tf_agents/trajectories/time_step/StepType.md#LAST"><code>StepType.LAST</code></a>.
