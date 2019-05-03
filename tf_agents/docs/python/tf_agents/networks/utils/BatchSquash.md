<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.networks.utils.BatchSquash" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="flatten"/>
<meta itemprop="property" content="unflatten"/>
</div>

# tf_agents.networks.utils.BatchSquash

## Class `BatchSquash`

Facilitates flattening and unflattening batch dims of a tensor.





Defined in [`networks/utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/networks/utils.py).

<!-- Placeholder for "Used in" -->

Exposes a pair of matched faltten and unflatten methods. After flattening only
1 batch dimension will be left. This facilitates evaluating networks that
expect inputs to have only 1 batch dimension.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(batch_dims)
```

Create two tied ops to flatten and unflatten the front dimensions.

#### Args:

* <b>`batch_dims`</b>: Number of batch dimensions the flatten/unflatten ops should
    handle.


#### Raises:

* <b>`ValueError`</b>: if batch dims is negative.



## Methods

<h3 id="flatten"><code>flatten</code></h3>

``` python
flatten(tensor)
```

Flattens and caches the tensor's batch_dims.

<h3 id="unflatten"><code>unflatten</code></h3>

``` python
unflatten(tensor)
```

Unflattens the tensor's batch_dims using the cached shape.



