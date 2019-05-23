# TF Agents Style Guide

This page contains style decisions that both developers and users of TF Agents
should follow to increase the readability of their code, reduce the
number of errors, and promote consistency.

## TensorFlow Style

Follow the [TensorFlow style guide](
https://www.tensorflow.org/community/style_guide) and
[documentation guide](https://www.tensorflow.org/community/documentation). Below
are additional TensorFlow conventions not noted in those guides. (In the future,
these noted conventions may be moved upstream.)

1.  The name is TensorFlow, not Tensorflow.

1.  Use `name_scope` at the beginning of every Python function.

    Justification: it’s easier to debug TF graphs when they align with Python
    code.

1.  Run all Tensor args through `tf.convert_to_tensor` immediately after
    name_scope.

    Justification: not doing so can lead to surprising results when computing
    gradients. It can also lead to unnecessary graph ops as subsequent TF calls
    will keep creating a tensor from the same op.

1.  Do not create new Tensors inside `@property` decorated methods.

    Justification: property requests cannot pass a `name=` argument to
    set a good name scope for these requests, making debugging harder.
    Furthermore, property requests should be lightweight and not perform
    too much computation.  But creating tensors, or even performing Eager
    computation, is relatively expensive compared to just accessing an existing
    property of the class.

1.  Every module should define the constant `__all__` in order to list all
    public members of the module.

    Justification: `__all__` is an explicit enumeration of what's intended to be
    public. It also governs what's imported when using `from foo import *`
    (although we cannot use star-import w/in Google, users can.) Use ticks for
    any Python objects, types, or code. E.g., write \`Tensor\` instead of
    Tensor.

1.  Do not use `tf.function` or `common.function` in library code, only in
    binary examples.

    Justification: When end-users want to debug eager mode, interjecting
    `tf.function` breaks their ability to do so.

1.  Use `utils.common.function` instead of `tf.function`.

    Justification: The default behavior of `tf.function` can be
    hard to debug (due to autograph, which we disable), and has serious
    performance issues with dynamically sized input arrays.

1.  Remove all calls to `tf.control_dependencies`.  Wrap
    `utils.common.function_in_tf1` around functions that use `tf.while_loop`,
    `tf.map_fn`, `tf.scan`, `tf.fold{l,r}`, and `tf.control_dependencies`.
    This will ensure that the new TF2-style control flow and TensorFlow's new
    "autodeps" work consistently across the codebase.  In TF2, all code is
    either Eager mode or wrapped in a `tf.function`, and in both cases you
    get autodeps for free.

    Justification: Code that uses old-style control flow (TF1 `tf.while_loop`
    outside of `tf.function`) and new-style control flow (TF2 `tf.while_loop`
    and TF1 `tf.while_loop` inside `tf.function`) does not mix.  We have moved
    most (all?) of TF Agents while loops to use the new-style control flow so we
    can have consistent behavior in TF1 and TF2.  Furthermore, TF currently
    has some bugs where overzealous `tf.control_dependencies`, when
    combined with `while_loop`, creates invalid graphs with loops.  Finally,
    autodeps will insert control deps as needed in the graph, avoiding
    overzealous control dependencies that cause a reduction in concurrency
    caused by the "shotgun" manual `tf.control_dependencies` approach.
