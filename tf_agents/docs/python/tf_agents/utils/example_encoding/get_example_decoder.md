<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.example_encoding.get_example_decoder" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.example_encoding.get_example_decoder

Get an example decoder function for a nested spec.

``` python
tf_agents.utils.example_encoding.get_example_decoder(
    example_spec,
    batched=False
)
```



Defined in [`utils/example_encoding.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/example_encoding.py).

<!-- Placeholder for "Used in" -->

Given a spec, returns an example decoder function. The decoder function parses
string serialized example protos into tensors according to the given spec.

#### Args:

* <b>`example_spec`</b>: list/tuple/nest of ArraySpecs describing a single example.
* <b>`batched`</b>: Boolean indicating if the decoder will receive batches of
    serialized data.


#### Returns:

Function

```python
decoder(serialized_proto: tf.tensor[string]) -> example_spec nest of tensors
```