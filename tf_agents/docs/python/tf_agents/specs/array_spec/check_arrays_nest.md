<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.array_spec.check_arrays_nest" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.specs.array_spec.check_arrays_nest

Check that the arrays conform to the spec.

``` python
tf_agents.specs.array_spec.check_arrays_nest(
    arrays,
    spec
)
```



Defined in [`specs/array_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/array_spec.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`arrays`</b>: A NumPy array, or a nested dict, list or tuple of arrays.
* <b>`spec`</b>: An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.


#### Returns:

True if the arrays conforms to the spec, False otherwise.