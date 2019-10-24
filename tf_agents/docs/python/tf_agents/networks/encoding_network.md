<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.encoding_network" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.networks.encoding_network

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/networks/encoding_network.py">View
source</a>

Keras Encoding Network.

<!-- Placeholder for "Used in" -->

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params

## Classes

[`class EncodingNetwork`](../../tf_agents/networks/encoding_network/EncodingNetwork.md):
Feed Forward network with CNN and FNN layers.
