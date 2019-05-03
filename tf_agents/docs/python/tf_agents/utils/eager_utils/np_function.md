<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils.np_function" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.eager_utils.np_function

Decorator that allow a numpy function to be used in Eager and Graph modes.

``` python
tf_agents.utils.eager_utils.np_function(
    func=None,
    output_dtypes=None
)
```



Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->

Similar to `tf.py_func` and `tf.py_function` but it doesn't require defining
the inputs or the dtypes of the outputs a priori.

In Eager mode it would convert the tf.Tensors to np.arrays before passing to
`func` and then convert back the outputs from np.arrays to tf.Tensors.

In Graph mode it would create different tf.py_function for each combination
of dtype of the inputs and cache them for reuse.

NOTE: In Graph mode: if `output_dtypes` is not provided then `func` would
be called with `np.ones()` to infer the output dtypes, and therefore `func`
should be stateless.

```python
Instead of doing:

def sum(x):
  return np.sum(x)
inputs = tf.constant([3, 4])
outputs = tf.py_function(sum, inputs, Tout=[tf.int64])

inputs = tf.constant([3., 4.])
outputs = tf.py_function(sum, inputs, Tout=[tf.float32])

Do:
@eager_utils.np_function
def sum(x):
  return np.sum(x)

inputs = tf.constant([3, 4])
outputs = sum(inputs)  # Infers that Tout is tf.int64

inputs = tf.constant([3., 4.])
outputs = sum(inputs)  # Infers that Tout is tf.float32

# Output dtype is always float32 for valid input dtypes.
@eager_utils.np_function(output_dtypes=np.float32)
def mean(x):
  return np.mean(x)

# Output dtype depends on the input dtype.
@eager_utils.np_function(output_dtypes=lambda x: (x, x))
def repeat(x):
  return x, x

with context.graph_mode():
  outputs = sum(tf.constant([3, 4]))
  outputs2 = sum(tf.constant([3., 4.]))
  sess.run(outputs) # np.array(7)
  sess.run(outputs2) # np.array(7.)

with context.eager_mode():
  inputs = tf.constant([3, 4])
  outputs = sum(tf.constant([3, 4])) # tf.Tensor([7])
  outputs = sum(tf.constant([3., 4.])) # tf.Tensor([7.])

```
#### Args:

* <b>`func`</b>: A numpy function, that takes numpy arrays as inputs and return numpy
    arrays as outputs.
* <b>`output_dtypes`</b>: Optional list of dtypes or a function that maps input dtypes
    to output dtypes. Examples: output_dtypes=[tf.float32],
    output_dtypes=lambda x: x (outputs have the same dtype as inputs).
    If it is not provided in Graph mode the `func` would be called to infer
    the output dtypes.

#### Returns:

A wrapped function that can be used with TF code.