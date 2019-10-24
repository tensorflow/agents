<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.value_rnn_network" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.networks.value_rnn_network

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/networks/value_rnn_network.py">View
source</a>

Sample Keras Value Network with LSTM cells .

<!-- Placeholder for "Used in" -->

Implements a network that will generate the following layers:

  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM   # lstm_cell_params
  [optional]: Dense  # output_fc_layer_params
  Dense -> 1         # Value output

## Classes

[`class ValueRnnNetwork`](../../tf_agents/networks/value_rnn_network/ValueRnnNetwork.md): Feed Forward value network. Reduces to 1 value output per batch item.

