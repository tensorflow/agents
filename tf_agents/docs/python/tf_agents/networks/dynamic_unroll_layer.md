<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.dynamic_unroll_layer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.networks.dynamic_unroll_layer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/networks/dynamic_unroll_layer.py">View
source</a>

Tensorflow RL Agent RNN utilities.

<!-- Placeholder for "Used in" -->

This module provides helper functions that Agents can use to train
RNN-based policies.

`DynamicUnroll`

The layer `DynamicUnroll` allows an Agent to train an RNN-based policy
by running an RNN over a batch of episode chunks from a replay buffer.

The agent creates a subclass of `tf.contrib.rnn.LayerRNNCell` or a Keras RNN
cell, such as `tf.keras.layers.LSTMCell`, instances of which
which can themselves be wrappers of `RNNCell`.  Training this instance
involes passing it to `DynamicUnroll` constructor; and then pass a set of
episode tensors in the form of `inputs`.

See the unit tests in `rnn_utils_test.py` for more details.

## Classes

[`class DynamicUnroll`](../../tf_agents/networks/dynamic_unroll_layer/DynamicUnroll.md): Process a history of sequences that are concatenated without padding.

