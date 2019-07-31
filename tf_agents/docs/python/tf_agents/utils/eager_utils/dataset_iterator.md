<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.dataset_iterator" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.eager_utils.dataset_iterator

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py">View
source</a>

Constructs a `Dataset` iterator.

``` python
tf_agents.utils.eager_utils.dataset_iterator(dataset)
```



<!-- Placeholder for "Used in" -->

The method used to construct the iterator is conditioned on whether Graph mode
is enabled. `dataset_iterator` and `get_next` are useful when we need to
construct an iterator and iterate through it inside a `tensorflow.function`.

#### Args:

* <b>`dataset`</b>: a `tf.data.Dataset`.

#### Returns:

A `tf.data.Iterator` if Graph mode is enabled; a tf.data.EagerIterator if in
eager mode.
