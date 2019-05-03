<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.lstm_encoding_network" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="KERAS_LSTM_FUSED_IMPLEMENTATION"/>
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.networks.lstm_encoding_network

Keras LSTM Encoding Network.



Defined in [`networks/lstm_encoding_network.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/networks/lstm_encoding_network.py).

<!-- Placeholder for "Used in" -->

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # input_conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM cell
  [optional]: Dense  # output_fc_layer_params

## Classes

[`class LSTMEncodingNetwork`](../../tf_agents/networks/lstm_encoding_network/LSTMEncodingNetwork.md): Recurrent network.

## Other Members

<h3 id="KERAS_LSTM_FUSED_IMPLEMENTATION"><code>KERAS_LSTM_FUSED_IMPLEMENTATION</code></h3>

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

