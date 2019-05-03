<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.utils.mlp_layers" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.networks.utils.mlp_layers

Generates conv and fc layers to encode into a hidden state.

``` python
tf_agents.networks.utils.mlp_layers(
    conv_layer_params=None,
    fc_layer_params=None,
    dropout_layer_params=None,
    activation_fn=tf.keras.activations.relu,
    kernel_initializer=None,
    name=None
)
```



Defined in [`networks/utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/networks/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`conv_layer_params`</b>: Optional list of convolution layers parameters, where
    each item is a length-three tuple indicating (filters, kernel_size,
    stride).
* <b>`fc_layer_params`</b>: Optional list of fully_connected parameters, where each
    item is the number of units in the layer.
* <b>`dropout_layer_params`</b>: Optional list of dropout layer parameters, each item
    is the fraction of input units to drop or a dictionary of parameters
    according to the keras.Dropout documentation. The additional parameter
    `permanent', if set to True, allows to apply dropout at inference for
    approximated Bayesian inference. The dropout layers are interleaved with
    the fully connected layers; there is a dropout layer after each fully
    connected layer, except if the entry in the list is None. This list must
    have the same length of fc_layer_params, or be None.
* <b>`activation_fn`</b>: Activation function, e.g. tf.keras.activations.relu,.
* <b>`kernel_initializer`</b>: Initializer to use for the kernels of the conv and
    dense layers. If none is provided a default variance_scaling_initializer
    is used.
* <b>`name`</b>: Name for the mlp layers.


#### Returns:

List of mlp layers.


#### Raises:

* <b>`ValueError`</b>: If the number of dropout layer parameters does not match the
    number of fully connected layer parameters.