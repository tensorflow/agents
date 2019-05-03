<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.example_encoding.get_example_encoder" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.example_encoding.get_example_encoder

Get example encoder function for the given spec.

``` python
tf_agents.utils.example_encoding.get_example_encoder(spec)
```



Defined in [`utils/example_encoding.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/example_encoding.py).

<!-- Placeholder for "Used in" -->

Given a spec, returns an example encoder function. The example encoder
function takes a nest of np.array feature values as input and returns a
TF Example proto.

Example:
  spec = {
      'lidar': array_spec.ArraySpec((900,), np.float32),
      'joint_positions': {
          'arm': array_spec.ArraySpec((7,), np.float32),
          'hand': array_spec.BoundedArraySpec((3, 3), np.int32, -1, 1)
      },
  }

  example_encoder = get_example_encoder(spec)
  serialized = example_encoder({
      'lidar': np.zeros((900,), np.float32),
      'joint_positions': {
          'arm': np.array([0.0, 1.57, 0.707, 0.2, 0.0, -1.57, 0.0],
                          np.float32),
          'hand': np.ones((3, 3), np.int32)
      },
  })

The returned example encoder function requires that the feature nest passed
has the shape and exact dtype specified in the spec. For example, it is
an error to pass an array with np.float64 dtype where np.float32 is expected.

#### Args:

* <b>`spec`</b>: list/tuple/nest of ArraySpecs describing a single example.


#### Returns:

Function

```python
encoder(features_nest of np.arrays) -> tf.train.Example
```