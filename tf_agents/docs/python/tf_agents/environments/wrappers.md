<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.environments.wrappers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.environments.wrappers

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/environments/wrappers.py">View
source</a>

Environment wrappers.

<!-- Placeholder for "Used in" -->

Wrappers in this module can be chained to change the overall behaviour of an
environment in common ways.

## Classes

[`class ActionClipWrapper`](../../tf_agents/environments/wrappers/ActionClipWrapper.md): Wraps an environment and clips actions to spec before applying.

[`class ActionDiscretizeWrapper`](../../tf_agents/environments/wrappers/ActionDiscretizeWrapper.md): Wraps an environment with continuous actions and discretizes them.

[`class ActionOffsetWrapper`](../../tf_agents/environments/wrappers/ActionOffsetWrapper.md): Offsets actions to be zero-based.

[`class ActionRepeat`](../../tf_agents/environments/wrappers/ActionRepeat.md): Repeates actions over n-steps while acummulating the received reward.

[`class FlattenObservationsWrapper`](../../tf_agents/environments/wrappers/FlattenObservationsWrapper.md): Wraps an environment and flattens nested multi-dimensional observations.

[`class GoalReplayEnvWrapper`](../../tf_agents/environments/wrappers/GoalReplayEnvWrapper.md): Adds a goal to the observation, used for HER (Hindsight Experience Replay).

[`class HistoryWrapper`](../../tf_agents/environments/wrappers/HistoryWrapper.md):
Adds observation and action history to the environment's observations.

[`class PyEnvironmentBaseWrapper`](../../tf_agents/environments/wrappers/PyEnvironmentBaseWrapper.md): PyEnvironment wrapper forwards calls to the given environment.

[`class RunStats`](../../tf_agents/environments/wrappers/RunStats.md): Wrapper that accumulates run statistics as the environment iterates.

[`class TimeLimit`](../../tf_agents/environments/wrappers/TimeLimit.md): End episodes after specified number of steps.

