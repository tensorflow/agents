<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.numpy_storage.NumpyState" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getattribute__"/>
<meta itemprop="property" content="__setattr__"/>
</div>

# tf_agents.utils.numpy_storage.NumpyState

## Class `NumpyState`

A checkpointable object whose NumPy array attributes are saved/restored.





Defined in [`utils/numpy_storage.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/numpy_storage.py).

<!-- Placeholder for "Used in" -->

Example usage:

```python
arrays = numpy_storage.NumpyState()
checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
arrays.x = np.ones([3, 4])
directory = self.get_temp_dir()
prefix = os.path.join(directory, 'ckpt')
save_path = checkpoint.save(prefix)
arrays.x[:] = 0.
assert (arrays.x == np.zeros([3, 4])).all()
checkpoint.restore(save_path)
assert (arrays.x == np.ones([3, 4])).all()

second_checkpoint = tf.train.Checkpoint(
    numpy_arrays=numpy_storage.NumpyState())
# Attributes of NumpyState objects are created automatically by restore()
second_checkpoint.restore(save_path)
assert (second_checkpoint.numpy_arrays.x == np.ones([3, 4])).all()
```

Note that `NumpyState` objects re-create the attributes of the previously
saved object on `restore()`. This is in contrast to TensorFlow variables, for
which a `Variable` object must be created and assigned to an attribute.

This snippet works both when graph building and when executing eagerly. On
save, the NumPy array(s) are fed as strings to be saved in the checkpoint (via
a placeholder when graph building, or as a string constant when executing
eagerly). When restoring they skip the TensorFlow graph entirely, and so no
restore ops need be run. This means that restoration always happens eagerly,
rather than waiting for `checkpoint.restore(...).run_restore_ops()` like
TensorFlow variables when graph building.

## Methods

<h3 id="__getattribute__"><code>__getattribute__</code></h3>

``` python
__getattribute__(name)
```

Un-wrap `_NumpyWrapper` objects when accessing attributes.

<h3 id="__setattr__"><code>__setattr__</code></h3>

``` python
__setattr__(
    name,
    value
)
```

Automatically wrap NumPy arrays assigned to attributes.



