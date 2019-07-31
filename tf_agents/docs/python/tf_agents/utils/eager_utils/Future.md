<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.Future" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf_agents.utils.eager_utils.Future

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py">View
source</a>

## Class `Future`

Converts a function or class method call into a future callable.



<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py">View
source</a>

``` python
__init__(
    func_or_method,
    *args,
    **kwargs
)
```

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py">View
source</a>

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
