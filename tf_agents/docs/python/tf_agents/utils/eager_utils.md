<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.eager_utils" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tf_agents.utils.eager_utils

Common utilities for TF-Agents.



Defined in [`utils/eager_utils.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/eager_utils.py).

<!-- Placeholder for "Used in" -->

Example of usage:

  ```python
  from tf_agents.utils import eager_utils

  @eager_utils.run_in_graph_and_eager_modes
  def loss_fn(x, y):
    v = tf.get_variable('v', initializer=tf.ones_initializer(), shape=())
    return v + x - y

  with tfe.graph_mode():
    # loss and train_step are Tensors/Ops in the graph
    loss_op = loss_fn(inputs, labels)
    train_step_op = eager_utils.create_train_step(loss_op, optimizer)
    # Compute the loss and apply gradients to the variables using the optimizer.
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(num_train_steps):
        loss_value = sess.run(train_step_op)

  with tfe.eager_mode():
    # loss and train_step are lambda functions that can be called.
    loss = loss_fn(inputs, labels)
    train_step = eager_utils.create_train_step(loss, optimizer)
    # Compute the loss and apply gradients to the variables using the optimizer.
    for _ in range(num_train_steps):
      loss_value = train_step()
  ```

## Classes

[`class Future`](../../tf_agents/utils/eager_utils/Future.md): Converts a function or class method call into a future callable.

## Functions

[`add_gradients_summaries(...)`](../../tf_agents/utils/eager_utils/add_gradients_summaries.md): Add summaries to gradients.

[`add_variables_summaries(...)`](../../tf_agents/utils/eager_utils/add_variables_summaries.md): Add summaries for variables.

[`clip_gradient_norms(...)`](../../tf_agents/utils/eager_utils/clip_gradient_norms.md): Clips the gradients by the given value.

[`clip_gradient_norms_fn(...)`](../../tf_agents/utils/eager_utils/clip_gradient_norms_fn.md): Returns a `transform_grads_fn` function for gradient clipping.

[`create_train_op(...)`](../../tf_agents/utils/eager_utils/create_train_op.md): Creates an `Operation` that evaluates the gradients and returns the loss.

[`create_train_step(...)`](../../tf_agents/utils/eager_utils/create_train_step.md): Creates a train_step that evaluates the gradients and returns the loss.

[`dataset_iterator(...)`](../../tf_agents/utils/eager_utils/dataset_iterator.md): Constructs a `Dataset` iterator.

[`future_in_eager_mode(...)`](../../tf_agents/utils/eager_utils/future_in_eager_mode.md): Decorator that allow a function/method to run in graph and in eager modes.

[`get_next(...)`](../../tf_agents/utils/eager_utils/get_next.md): Returns the next element in a `Dataset` iterator.

[`has_self_cls_arg(...)`](../../tf_agents/utils/eager_utils/has_self_cls_arg.md): Checks if it is method which takes self/cls as the first argument.

[`is_unbound(...)`](../../tf_agents/utils/eager_utils/is_unbound.md): Checks if it is an unbounded method.

[`np_function(...)`](../../tf_agents/utils/eager_utils/np_function.md): Decorator that allow a numpy function to be used in Eager and Graph modes.

## Other Members

<h3 id="absolute_import"><code>absolute_import</code></h3>

<h3 id="division"><code>division</code></h3>

<h3 id="print_function"><code>print_function</code></h3>

