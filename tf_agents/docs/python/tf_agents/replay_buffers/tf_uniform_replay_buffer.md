<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.tf_uniform_replay_buffer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.replay_buffers.tf_uniform_replay_buffer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/tf_uniform_replay_buffer.py">View
source</a>

A batched replay buffer of nests of Tensors which can be sampled uniformly.

<!-- Placeholder for "Used in" -->

- Each add assumes tensors have batch_size as first dimension, and will store
each element of the batch in an offset segment, so that each batch dimension has
its own contiguous memory. Within batch segments, behaves as a circular buffer.

The get_next function returns 'ids' in addition to the data. This is not really
needed for the batched replay buffer, but is returned to be consistent with
the API for a priority replay buffer, which needs the ids to update priorities.

## Classes

[`class BufferInfo`](../../tf_agents/replay_buffers/tf_uniform_replay_buffer/BufferInfo.md): BufferInfo(ids, probabilities)

[`class TFUniformReplayBuffer`](../../tf_agents/replay_buffers/tf_uniform_replay_buffer/TFUniformReplayBuffer.md): A TFUniformReplayBuffer with batched adds and uniform sampling.

