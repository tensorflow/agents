<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.nest_utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.utils.nest_utils

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/nest_utils.py">View
source</a>

Utilities for handling nested tensors.

<!-- Placeholder for "Used in" -->


## Functions

[`batch_nested_array(...)`](../../tf_agents/utils/nest_utils/batch_nested_array.md)

[`batch_nested_tensors(...)`](../../tf_agents/utils/nest_utils/batch_nested_tensors.md): Add batch dimension if needed to nested tensors while checking their specs.

[`fast_map_structure(...)`](../../tf_agents/utils/nest_utils/fast_map_structure.md)

[`fast_map_structure_flatten(...)`](../../tf_agents/utils/nest_utils/fast_map_structure_flatten.md)

[`flatten_multi_batched_nested_tensors(...)`](../../tf_agents/utils/nest_utils/flatten_multi_batched_nested_tensors.md): Reshape tensors to contain only one batch dimension.

[`flatten_with_joined_paths(...)`](../../tf_agents/utils/nest_utils/flatten_with_joined_paths.md)

[`get_outer_array_shape(...)`](../../tf_agents/utils/nest_utils/get_outer_array_shape.md): Batch dims of array's batch dimension `dim`.

[`get_outer_rank(...)`](../../tf_agents/utils/nest_utils/get_outer_rank.md): Compares tensors to specs to determine the number of batch dimensions.

[`get_outer_shape(...)`](../../tf_agents/utils/nest_utils/get_outer_shape.md): Runtime batch dims of tensor's batch dimension `dim`.

[`has_tensors(...)`](../../tf_agents/utils/nest_utils/has_tensors.md)

[`is_batched_nested_tensors(...)`](../../tf_agents/utils/nest_utils/is_batched_nested_tensors.md): Compares tensors to specs to determine if all tensors are batched or not.

[`split_nested_tensors(...)`](../../tf_agents/utils/nest_utils/split_nested_tensors.md): Split batched nested tensors, on batch dim (outer dim), into a list.

[`stack_nested_arrays(...)`](../../tf_agents/utils/nest_utils/stack_nested_arrays.md): Stack/batch a list of nested numpy arrays.

[`stack_nested_tensors(...)`](../../tf_agents/utils/nest_utils/stack_nested_tensors.md): Stacks a list of nested tensors along the first dimension.

[`unbatch_nested_array(...)`](../../tf_agents/utils/nest_utils/unbatch_nested_array.md)

[`unbatch_nested_tensors(...)`](../../tf_agents/utils/nest_utils/unbatch_nested_tensors.md): Remove the batch dimension if needed from nested tensors using their specs.

[`unstack_nested_arrays(...)`](../../tf_agents/utils/nest_utils/unstack_nested_arrays.md): Unstack/unbatch a nest of numpy arrays.

[`unstack_nested_tensors(...)`](../../tf_agents/utils/nest_utils/unstack_nested_tensors.md): Make list of unstacked nested tensors.

[`where(...)`](../../tf_agents/utils/nest_utils/where.md): Generalization of
tf.compat.v1.where supporting nests as the outputs.
