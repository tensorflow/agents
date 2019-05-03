<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.metrics.py_metrics.NumpyDeque" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getattribute__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="extend"/>
<meta itemprop="property" content="mean"/>
</div>

# tf_agents.metrics.py_metrics.NumpyDeque

## Class `NumpyDeque`

Deque implementation using a numpy array as a circular buffer.

Inherits From: [`NumpyState`](../../../tf_agents/utils/numpy_storage/NumpyState.md)



Defined in [`metrics/py_metrics.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    maxlen,
    dtype
)
```

Deque using a numpy array as a circular buffer, with FIFO evictions.

#### Args:

* <b>`maxlen`</b>: Maximum length of the deque before beginning to evict the oldest
    entries. If np.inf, deque size is unlimited and the array will grow
    automatically.
* <b>`dtype`</b>: Data type of deque elements.



## Methods

<h3 id="__getattribute__"><code>__getattribute__</code></h3>

``` python
__getattribute__(name)
```

Un-wrap `_NumpyWrapper` objects when accessing attributes.

<h3 id="__len__"><code>__len__</code></h3>

``` python
__len__()
```



<h3 id="__setattr__"><code>__setattr__</code></h3>

``` python
__setattr__(
    name,
    value
)
```

Automatically wrap NumPy arrays assigned to attributes.

<h3 id="add"><code>add</code></h3>

``` python
add(value)
```



<h3 id="clear"><code>clear</code></h3>

``` python
clear()
```



<h3 id="extend"><code>extend</code></h3>

``` python
extend(values)
```



<h3 id="mean"><code>mean</code></h3>

``` python
mean(dtype=None)
```





