<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.numpy_storage.NumpyState" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.numpy_storage.NumpyState

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/numpy_storage.py">View
source</a>

## Class `NumpyState`

A checkpointable object whose NumPy array attributes are saved/restored.

<!-- Placeholder for "Used in" -->

#### Example usage:

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

