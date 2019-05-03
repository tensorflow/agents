<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.function_in_tf1" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.function_in_tf1

Wrapper that returns common.function if using TF1.

``` python
tf_agents.utils.common.function_in_tf1(
    *args,
    **kwargs
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

This allows for code that assumes autodeps is available to be written once,
in the same way, for both TF1 and TF2.

Usage:

```python
train = function_in_tf1()(agent.train)
loss = train(experience)
```

#### Args:

* <b>`*args`</b>: Arguments for common.function.
* <b>`**kwargs`</b>: Keyword arguments for common.function.


#### Returns:

A callable that wraps a function.