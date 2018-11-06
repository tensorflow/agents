# TF Agents Style Guide

This page contains style decisions that both developers and users of TF Agents
should follow to increase the readability of their code, reduce the
number of errors, and promote consistency.

## TensorFlow Style

Follow the [TensorFlow style
guide](https://www.tensorflow.org/community/style_guide) and [documentation
guide](https://www.tensorflow.org/community/documentation). Below are additional
TensorFlow conventions not noted in those guides. (In the future, these noted
conventions may be moved upstream.)

1.  The name is TensorFlow, not Tensorflow.
1.  Use `name_scope` at the beginning of every Python function.

    Justification: it’s easier to debug TF graphs when they align with Python
    code.

1.  Run all Tensor args through `tf.convert_to_tensor` immediately after
    name_scope.

    Justification: not doing so can lead to surprising results when computing
    gradients. It can also lead to unnecessary graph ops as subsequent TF calls
    will keep creating a tensor from the same op.

1.  Every module should define the constant `__all__` in order to list all
    public members of the module.

    Justification: `__all__` is an explicit enumeration of what's intended to be
    public. It also governs what's imported when using `from foo import *`
    (although we cannot use star-import w/in Google, users can.) Use ticks for
    any Python objects, types, or code. E.g., write \`Tensor\` instead of
    Tensor.
