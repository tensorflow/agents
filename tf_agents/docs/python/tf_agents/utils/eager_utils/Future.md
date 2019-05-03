<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.Future" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf_agents.utils.eager_utils.Future

## Class `Future`

Converts a function or class method call into a future callable.





Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    func_or_method,
    *args,
    **kwargs
)
```





## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    *args,
    **kwargs
)
```

If *args/**kwargs are given they would replace those given at init.

#### Args:

* <b>`*args`</b>: List of extra arguments.
* <b>`**kwargs`</b>: Dict of extra keyword arguments.

#### Returns:

The result of func_or_method(*args, **kwargs).



