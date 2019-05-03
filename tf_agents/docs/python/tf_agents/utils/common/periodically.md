<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.periodically" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.periodically

Periodically performs the tensorflow op in `body`.

``` python
tf_agents.utils.common.periodically(
    body,
    period,
    name='periodically'
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

The body tensorflow op will be executed every `period` times the periodically
op is executed. More specifically, with `n` the number of times the op has
been executed, the body will be executed when `n` is a non zero positive
multiple of `period` (i.e. there exist an integer `k > 0` such that
`k * period == n`).

If `period` is `None`, it will not perform any op and will return a
`tf.no_op()`.

If `period` is 1, it will just execute the body, and not create any counters
or conditionals.

#### Args:

* <b>`body`</b>: callable that returns the tensorflow op to be performed every time
    an internal counter is divisible by the period. The op must have no
    output (for example, a tf.group()).
* <b>`period`</b>: inverse frequency with which to perform the op.
* <b>`name`</b>: name of the variable_scope.


#### Raises:

* <b>`TypeError`</b>: if body is not a callable.


#### Returns:

An op that periodically performs the specified op.