<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.trajectories.trajectory" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.trajectories.trajectory

Trajectory containing time_step transition information.



Defined in [`trajectories/trajectory.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/trajectories/trajectory.py).

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

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

