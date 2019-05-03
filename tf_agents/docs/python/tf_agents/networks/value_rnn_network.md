<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.value_rnn_network" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.networks.value_rnn_network

Sample Keras Value Network with LSTM cells .



Defined in [`networks/value_rnn_network.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/networks/value_rnn_network.py).

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

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

