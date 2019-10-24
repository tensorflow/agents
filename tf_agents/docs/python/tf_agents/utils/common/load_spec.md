<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.load_spec" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.load_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Loads a data spec from a file.

```python
tf_agents.utils.common.load_spec(file_path)
```

<!-- Placeholder for "Used in" -->

**Note**: Types for Named tuple classes will not match. Users need to convert to
these manually:

\# Convert from: #
'tensorflow.python.saved_model.nested_structure_coder.Trajectory' # to proper
TrajectorySpec. # trajectory_spec = trajectory.Trajectory(*spec)

#### Args:

*   <b>`file_path`</b>: Path to the saved data spec.

#### Returns:

A nested structure of TensorSpecs.
