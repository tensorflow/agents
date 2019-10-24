<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.array_spec" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.specs.array_spec

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/specs/array_spec.py">View
source</a>

A class to describe the shape and dtype of numpy arrays.

<!-- Placeholder for "Used in" -->


## Classes

[`class ArraySpec`](../../tf_agents/specs/ArraySpec.md): Describes a numpy array or scalar shape and dtype.

[`class BoundedArraySpec`](../../tf_agents/specs/BoundedArraySpec.md): An `ArraySpec` that specifies minimum and maximum values.

## Functions

[`add_outer_dims_nest(...)`](../../tf_agents/specs/array_spec/add_outer_dims_nest.md)

[`check_arrays_nest(...)`](../../tf_agents/specs/array_spec/check_arrays_nest.md): Check that the arrays conform to the spec.

[`is_bounded(...)`](../../tf_agents/specs/array_spec/is_bounded.md)

[`is_continuous(...)`](../../tf_agents/specs/array_spec/is_continuous.md)

[`is_discrete(...)`](../../tf_agents/specs/array_spec/is_discrete.md)

[`sample_bounded_spec(...)`](../../tf_agents/specs/array_spec/sample_bounded_spec.md): Samples the given bounded spec.

[`sample_spec_nest(...)`](../../tf_agents/specs/array_spec/sample_spec_nest.md): Samples the given nest of specs.

[`update_spec_shape(...)`](../../tf_agents/specs/array_spec/update_spec_shape.md):
Returns a copy of the given spec with the new shape.
