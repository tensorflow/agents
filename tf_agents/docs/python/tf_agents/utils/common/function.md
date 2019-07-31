<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.function" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.function

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Wrapper for tf.function with TF Agents-specific customizations.

``` python
tf_agents.utils.common.function(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

#### Example:

```python
@common.function()
def my_eager_code(x, y):
  ...
```

#### Args:

* <b>`*args`</b>: Args for tf.function.
* <b>`**kwargs`</b>: Keyword args for tf.function.


#### Returns:

A tf.function wrapper.
