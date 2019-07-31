<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.save_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.save_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Saves the given spec nest as a StructProto.

```python
tf_agents.utils.common.save_spec(
    spec,
    file_path
)
```

<!-- Placeholder for "Used in" -->

**Note**: Currently this will convert BoundedTensorSpecs into regular
TensorSpecs.

#### Args:

*   <b>`spec`</b>: A nested structure of TensorSpecs.
*   <b>`file_path`</b>: Path to save the encoded spec to.
