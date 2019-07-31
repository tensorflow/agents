<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.py_uniform_replay_buffer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.replay_buffers.py_uniform_replay_buffer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/py_uniform_replay_buffer.py">View
source</a>

Uniform replay buffer in Python.

<!-- Placeholder for "Used in" -->

The base class provides all the functionalities of a uniform replay buffer:
  - add samples in a First In First Out way.
  - read samples uniformly.

PyHashedReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).

## Classes

[`class PyUniformReplayBuffer`](../../tf_agents/replay_buffers/py_uniform_replay_buffer/PyUniformReplayBuffer.md): A Python-based replay buffer that supports uniform sampling.

