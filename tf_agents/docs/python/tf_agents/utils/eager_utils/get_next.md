<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.get_next" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.eager_utils.get_next

Returns the next element in a `Dataset` iterator.

``` python
tf_agents.utils.eager_utils.get_next(iterator)
```



Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->

The syntax used to retrieve the next item is conditioned on whether Graph mode
is enabled. `dataset_iterator` and `get_next` are useful when we need to
construct an iterator and iterate through it inside a `tensorflow.function`.

#### Args:

* <b>`iterator`</b>: a `tf.data.Iterator` if in Graph mode; a `tf.data.EagerIterator`
    if in eager mode.

#### Returns:

A `tf.data.Iterator` if Graph mode is enabled; a Python iterator if in eager
mode.