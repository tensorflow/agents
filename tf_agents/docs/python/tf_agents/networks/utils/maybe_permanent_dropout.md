<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.utils.maybe_permanent_dropout" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.networks.utils.maybe_permanent_dropout

Adds a Keras dropout layer with the option of applying it at inference.

``` python
tf_agents.networks.utils.maybe_permanent_dropout(
    rate,
    noise_shape=None,
    seed=None,
    permanent=False
)
```



Defined in [`networks/utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/networks/utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`rate`</b>: the probability of dropping an input.
* <b>`noise_shape`</b>: 1D integer tensor representing the dropout mask multiplied to
    the input.
* <b>`seed`</b>: A Python integer to use as random seed.
* <b>`permanent`</b>: If set, applies dropout during inference and not only during
    training. This flag is used for approximated Bayesian inference.

#### Returns:

A function adding a dropout layer according to the parameters for the given
  input.