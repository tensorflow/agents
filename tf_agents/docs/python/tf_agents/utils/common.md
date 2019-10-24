<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="MISSING_RESOURCE_VARIABLES_ERROR"/>
</div>

# Module: tf_agents.utils.common

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Common utilities for TF-Agents.

<!-- Placeholder for "Used in" -->


## Classes

[`class Checkpointer`](../../tf_agents/utils/common/Checkpointer.md): Checkpoints training state, policy state, and replay_buffer state.

[`class EagerPeriodically`](../../tf_agents/utils/common/EagerPeriodically.md): EagerPeriodically performs the ops defined in `body`.

[`class OUProcess`](../../tf_agents/utils/common/OUProcess.md): A zero-mean Ornstein-Uhlenbeck process.

[`class Periodically`](../../tf_agents/utils/common/Periodically.md): Periodically performs the ops defined in `body`.

## Functions

[`assert_members_are_not_overridden(...)`](../../tf_agents/utils/common/assert_members_are_not_overridden.md): Asserts public members of `base_cls` are not overridden in `instance`.

[`clip_to_spec(...)`](../../tf_agents/utils/common/clip_to_spec.md): Clips value to a given bounded tensor spec.

[`compute_returns(...)`](../../tf_agents/utils/common/compute_returns.md): Compute the return from each index in an episode.

[`convert_q_logits_to_values(...)`](../../tf_agents/utils/common/convert_q_logits_to_values.md): Converts a set of Q-value logits into Q-values using the provided support.

[`create_variable(...)`](../../tf_agents/utils/common/create_variable.md): Create a variable.

[`discounted_future_sum(...)`](../../tf_agents/utils/common/discounted_future_sum.md): Discounted future sum of batch-major values.

[`discounted_future_sum_masked(...)`](../../tf_agents/utils/common/discounted_future_sum_masked.md): Discounted future sum of batch-major values.

[`element_wise_huber_loss(...)`](../../tf_agents/utils/common/element_wise_huber_loss.md)

[`element_wise_squared_loss(...)`](../../tf_agents/utils/common/element_wise_squared_loss.md)

[`entropy(...)`](../../tf_agents/utils/common/entropy.md): Computes total entropy of distribution.

[`function(...)`](../../tf_agents/utils/common/function.md): Wrapper for tf.function with TF Agents-specific customizations.

[`function_in_tf1(...)`](../../tf_agents/utils/common/function_in_tf1.md): Wrapper that returns common.function if using TF1.

[`generate_tensor_summaries(...)`](../../tf_agents/utils/common/generate_tensor_summaries.md): Generates various summaries of `tensor` such as histogram, max, min, etc.

[`get_contiguous_sub_episodes(...)`](../../tf_agents/utils/common/get_contiguous_sub_episodes.md): Computes mask on sub-episodes which includes only contiguous components.

[`get_episode_mask(...)`](../../tf_agents/utils/common/get_episode_mask.md): Create a mask that is 0.0 for all final steps, 1.0 elsewhere.

[`has_eager_been_enabled(...)`](../../tf_agents/utils/common/has_eager_been_enabled.md): Returns true iff in TF2 or in TF1 with eager execution enabled.

[`index_with_actions(...)`](../../tf_agents/utils/common/index_with_actions.md): Index into q_values using actions.

[`initialize_uninitialized_variables(...)`](../../tf_agents/utils/common/initialize_uninitialized_variables.md): Initialize any pending variables that are uninitialized.

[`join_scope(...)`](../../tf_agents/utils/common/join_scope.md): Joins a parent and child scope using `/`, checking for empty/none.

[`load_spec(...)`](../../tf_agents/utils/common/load_spec.md): Loads a data spec
from a file.

[`log_probability(...)`](../../tf_agents/utils/common/log_probability.md): Computes log probability of actions given distribution.

[`ornstein_uhlenbeck_process(...)`](../../tf_agents/utils/common/ornstein_uhlenbeck_process.md): An op for generating noise from a zero-mean Ornstein-Uhlenbeck process.

[`periodically(...)`](../../tf_agents/utils/common/periodically.md): Periodically performs the tensorflow op in `body`.

[`replicate(...)`](../../tf_agents/utils/common/replicate.md): Replicates a tensor so as to match the given outer shape.

[`resource_variables_enabled(...)`](../../tf_agents/utils/common/resource_variables_enabled.md)

[`save_spec(...)`](../../tf_agents/utils/common/save_spec.md): Saves the given
spec nest as a StructProto.

[`scale_to_spec(...)`](../../tf_agents/utils/common/scale_to_spec.md): Shapes and scales a batch into the given spec bounds.

[`shift_values(...)`](../../tf_agents/utils/common/shift_values.md): Shifts batch-major values in time by some amount.

[`soft_variables_update(...)`](../../tf_agents/utils/common/soft_variables_update.md): Performs a soft/hard update of variables from the source to the target.

[`spec_means_and_magnitudes(...)`](../../tf_agents/utils/common/spec_means_and_magnitudes.md): Get the center and magnitude of the ranges in action spec.

[`transpose_batch_time(...)`](../../tf_agents/utils/common/transpose_batch_time.md): Transposes the batch and time dimensions of a Tensor.

## Other Members

*   `MISSING_RESOURCE_VARIABLES_ERROR = '\nResource variables are not enabled.
    Please enable them by adding the following\ncode to your main() method:\n
    tf.compat.v1.enable_resource_variables()\nFor unit tests,
    subclass`tf_agents.utils.test_utils.TestCase`.\n'`
    <a id="MISSING_RESOURCE_VARIABLES_ERROR"></a>
