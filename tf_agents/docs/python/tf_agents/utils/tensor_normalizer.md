<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.tensor_normalizer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_agents.utils.tensor_normalizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/tensor_normalizer.py">View
source</a>

Tensor normalizer classses.

<!-- Placeholder for "Used in" -->

These encapsulate variables and function for tensor normalization.

#### Example usage:

observation = tf.placeholder(tf.float32, shape=[])
tensor_normalizer = StreamingTensorNormalizer(
    tensor_spec.TensorSpec([], tf.float32), scope='normalize_observation')
normalized_observation = tensor_normalizer.normalize(observation)
update_normalization = tensor_normalizer.update(observation)

with tf.Session() as sess:
  for o in observation_list:
    # Compute normalized observation given current observation vars.
    normalized_observation_ = sess.run(
        normalized_observation, feed_dict = {observation: o})

    # Update normalization params for next normalization op.
    sess.run(update_normalization, feed_dict = {observation: o})

    # Do something with normalized_observation_
    ...

## Classes

[`class EMATensorNormalizer`](../../tf_agents/utils/tensor_normalizer/EMATensorNormalizer.md): TensorNormalizer with exponential moving avg. mean and var estimates.

[`class StreamingTensorNormalizer`](../../tf_agents/utils/tensor_normalizer/StreamingTensorNormalizer.md): Normalizes mean & variance based on full history of tensor values.

[`class TensorNormalizer`](../../tf_agents/utils/tensor_normalizer/TensorNormalizer.md): Encapsulates tensor normalization and owns normalization variables.

