<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.encoding_network" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.networks.encoding_network

Keras Encoding Network.



Defined in [`networks/encoding_network.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/networks/encoding_network.py).

<!-- Placeholder for "Used in" -->

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params

## Classes

[`class EncodingNetwork`](../../tf_agents/networks/encoding_network/EncodingNetwork.md): Feed Forward network with CNN and FNN layers..

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

