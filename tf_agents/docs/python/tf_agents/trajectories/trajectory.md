<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.trajectories.trajectory

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py">View
source</a>

Trajectory containing time_step transition information.

<!-- Placeholder for "Used in" -->


## Classes

[`class Trajectory`](../../tf_agents/trajectories/trajectory/Trajectory.md): A tuple that represents a trajectory.

## Functions

[`boundary(...)`](../../tf_agents/trajectories/trajectory/boundary.md): Create a Trajectory transitioning between StepTypes `LAST` and `FIRST`.

[`first(...)`](../../tf_agents/trajectories/trajectory/first.md): Create a Trajectory transitioning between StepTypes `FIRST` and `MID`.

[`from_episode(...)`](../../tf_agents/trajectories/trajectory/from_episode.md): Create a Trajectory from tensors representing a single episode.

[`from_transition(...)`](../../tf_agents/trajectories/trajectory/from_transition.md): Returns a `Trajectory` given transitions.

[`last(...)`](../../tf_agents/trajectories/trajectory/last.md): Create a Trajectory transitioning between StepTypes `MID` and `LAST`.

[`mid(...)`](../../tf_agents/trajectories/trajectory/mid.md): Create a Trajectory transitioning between StepTypes `MID` and `MID`.

[`to_transition(...)`](../../tf_agents/trajectories/trajectory/to_transition.md): Create a transition from a trajectory or two adjacent trajectories.

[`to_transition_spec(...)`](../../tf_agents/trajectories/trajectory/to_transition_spec.md):
Create a transition spec from a trajectory spec.
