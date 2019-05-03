<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.py_hashed_replay_buffer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.replay_buffers.py_hashed_replay_buffer

Uniform replay buffer in Python with compressed storage.



Defined in [`replay_buffers/py_hashed_replay_buffer.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/py_hashed_replay_buffer.py).

<!-- Placeholder for "Used in" -->

PyHashedReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).

## Classes

[`class FrameBuffer`](../../tf_agents/replay_buffers/py_hashed_replay_buffer/FrameBuffer.md): Saves some frames in a memory efficient way.

[`class PyHashedReplayBuffer`](../../tf_agents/replay_buffers/py_hashed_replay_buffer/PyHashedReplayBuffer.md): A Python-based replay buffer with optimized underlying storage.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

