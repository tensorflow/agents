<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.future_in_eager_mode" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.eager_utils.future_in_eager_mode

Decorator that allow a function/method to run in graph and in eager modes.

``` python
tf_agents.utils.eager_utils.future_in_eager_mode(func_or_method)
```



Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->

When applied in graph mode it calls the function and return its outputs.
When applied in eager mode it returns a lambda function that when called
returns the outputs.

```python
@eager_utils.future_in_eager_mode
def loss_fn(x):
  v = tf.get_variable('v', initializer=tf.ones_initializer(), shape=())
  return v + x

with context.graph_mode():
  loss_op = loss_fn(inputs)
  loss_value = sess.run(loss_op)

with context.eager_mode():
  loss = loss_fn(inputs)
  # Now loss is a Future callable.
  loss_value = loss()

#### Args:

* <b>`func_or_method`</b>: A function or method to decorate.


#### Returns:

Either the output ops of the function/method or a Future (lambda function).