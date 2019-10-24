<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.py_hashed_replay_buffer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.replay_buffers.py_hashed_replay_buffer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/py_hashed_replay_buffer.py">View
source</a>

Uniform replay buffer in Python with compressed storage.

<!-- Placeholder for "Used in" -->

PyHashedReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).

## Classes

[`class FrameBuffer`](../../tf_agents/replay_buffers/py_hashed_replay_buffer/FrameBuffer.md): Saves some frames in a memory efficient way.

[`class PyHashedReplayBuffer`](../../tf_agents/replay_buffers/py_hashed_replay_buffer/PyHashedReplayBuffer.md): A Python-based replay buffer with optimized underlying storage.

