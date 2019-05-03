<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.EagerPeriodically" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf_agents.utils.common.EagerPeriodically

## Class `EagerPeriodically`

EagerPeriodically performs the ops defined in `body`.





Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

Only works in Eager mode.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    body,
    period
)
```

EagerPeriodically performs the ops defined in `body`.

#### Args:

* <b>`body`</b>: callable that returns the tensorflow op to be performed every time
    an internal counter is divisible by the period. The op must have no
    output (for example, a tf.group()).
* <b>`period`</b>: inverse frequency with which to perform the op.
    Must be a simple python int/long.


#### Raises:

* <b>`TypeError`</b>: if body is not a callable.


#### Returns:

An op that periodically performs the specified op.



## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__()
```





