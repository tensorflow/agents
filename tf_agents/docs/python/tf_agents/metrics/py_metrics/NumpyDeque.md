<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.metrics.py_metrics.NumpyDeque" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="extend"/>
<meta itemprop="property" content="mean"/>
</div>

# tf_agents.metrics.py_metrics.NumpyDeque

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

## Class `NumpyDeque`

Deque implementation using a numpy array as a circular buffer.

Inherits From: [`NumpyState`](../../../tf_agents/utils/numpy_storage/NumpyState.md)

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

``` python
__init__(
    maxlen,
    dtype
)
```

Deque using a numpy array as a circular buffer, with FIFO evictions.

#### Args:

*   <b>`maxlen`</b>: Maximum length of the deque before beginning to evict the
    oldest entries. If np.inf, deque size is unlimited and the array will grow
    automatically.
*   <b>`dtype`</b>: Data type of deque elements.

## Methods

<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

``` python
__len__()
```




<h3 id="add"><code>add</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

``` python
add(value)
```

<h3 id="clear"><code>clear</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

``` python
clear()
```

<h3 id="extend"><code>extend</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

``` python
extend(values)
```

<h3 id="mean"><code>mean</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metrics.py">View
source</a>

``` python
mean(dtype=None)
```
