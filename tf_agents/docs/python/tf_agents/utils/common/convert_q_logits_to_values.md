<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.convert_q_logits_to_values" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.convert_q_logits_to_values

Converts a set of Q-value logits into Q-values using the provided support.

``` python
tf_agents.utils.common.convert_q_logits_to_values(
    logits,
    support
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`logits`</b>: A Tensor representing the Q-value logits.
* <b>`support`</b>: The support of the underlying distribution.


#### Returns:

A Tensor containing the expected Q-values.