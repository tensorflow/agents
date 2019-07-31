<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.lstm_encoding_network" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="KERAS_LSTM_FUSED_IMPLEMENTATION"/>
</div>

# Module: tf_agents.networks.lstm_encoding_network

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/networks/lstm_encoding_network.py">View
source</a>

Keras LSTM Encoding Network.

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

*   `KERAS_LSTM_FUSED_IMPLEMENTATION = 2`
    <a id="KERAS_LSTM_FUSED_IMPLEMENTATION"></a>
