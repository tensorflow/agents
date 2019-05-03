<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.py_uniform_replay_buffer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.replay_buffers.py_uniform_replay_buffer

Uniform replay buffer in Python.



Defined in [`replay_buffers/py_uniform_replay_buffer.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/py_uniform_replay_buffer.py).

<!-- Placeholder for "Used in" -->

The base class provides all the functionalities of a uniform replay buffer:
  - add samples in a First In First Out way.
  - read samples uniformly.

PyHashedReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).

## Classes

[`class PyUniformReplayBuffer`](../../tf_agents/replay_buffers/py_uniform_replay_buffer/PyUniformReplayBuffer.md): A Python-based replay buffer that supports uniform sampling.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

