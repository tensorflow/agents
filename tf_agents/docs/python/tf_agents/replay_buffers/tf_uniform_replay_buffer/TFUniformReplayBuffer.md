<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="capacity"/>
<meta itemprop="property" content="data_spec"/>
<meta itemprop="property" content="device"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="scope"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="table_fn"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_batch"/>
<meta itemprop="property" content="as_dataset"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="gather_all"/>
<meta itemprop="property" content="get_next"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/tf_uniform_replay_buffer.py">View
source</a>

## Class `TFUniformReplayBuffer`

A TFUniformReplayBuffer with batched adds and uniform sampling.

Inherits From: [`ReplayBuffer`](../../../tf_agents/replay_buffers/replay_buffer/ReplayBuffer.md)

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Creates a TFUniformReplayBuffer.

The TFUniformReplayBuffer stores episodes in `B == batch_size` blocks of size
`L == max_length`, with total frame capacity `C == L * B`. Storage looks like:

```
block1 ep1 frame1
           frame2
       ...
       ep2 frame1
           frame2
       ...
       <L frames total>
block2 ep1 frame1
           frame2
       ...
       ep2 frame1
           frame2
       ...
       <L frames total>
...
blockB ep1 frame1
           frame2
       ...
       ep2 frame1
           frame2
       ...
       <L frames total>
```

Multiple episodes may be stored within a given block, up to `max_length` frames
total. In practice, new episodes will overwrite old ones as the block rolls over
its `max_length`.

#### Args:

*   <b>`data_spec`</b>: A TensorSpec or a list/tuple/nest of TensorSpecs
    describing a single item that can be stored in this buffer.
*   <b>`batch_size`</b>: Batch dimension of tensors when adding to buffer.
*   <b>`max_length`</b>: The maximum number of items that can be stored in a
    single batch segment of the buffer.
*   <b>`scope`</b>: Scope prefix for variables and ops created by this class.
*   <b>`device`</b>: A TensorFlow device to place the Variables and ops.
*   <b>`table_fn`</b>: Function to create tables `table_fn(data_spec, capacity)`
    that can read/write nested tensors.
*   <b>`dataset_drop_remainder`</b>: If `True`, then when calling `as_dataset`
    with arguments `single_deterministic_pass=True` and `sample_batch_size !=
    None`, the final batch will be dropped if it does not contain exactly
    `sample_batch_size` items. This is helpful for static shape inference as the
    resulting tensors will always have leading dimension `sample_batch_size`
    instead of `None`.
*   <b>`dataset_window_shift`</b>: Window shift used when calling `as_dataset`
    with arguments `single_deterministic_pass=True` and `num_steps != None`.
    This determines how the resulting frames are windowed. If `None`, then there
    is no overlap created between frames and each frame is seen exactly once.
    For example, if `max_length=5`, `num_steps=2`, `sample_batch_size=None`, and
    `dataset_window_shift=None`, then the datasets returned will have frames
    `{[0, 1], [2, 3], [4]}`.

    If `num_steps != None`, then windows are created with a window overlap of
    `dataset_window_shift` and you will see each frame up to `num_steps` times.
    For example, if `max_length=5`, `num_steps=2`, `sample_batch_size=None`, and
    `dataset_window_shift=1`, then the datasets returned will have windows of
    shifted repeated frames: `{[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]}`.

    For more details, see the documentation of `tf.data.Dataset.window`,
    specifically for the `shift` argument.

    The default behavior is to not overlap frames (`dataset_window_shift=None`)
    but users often want to see all combinations of frame sequences, in which
    case `dataset_window_shift=1` is the appropriate value.

## Properties

<h3 id="capacity"><code>capacity</code></h3>

Returns the capacity of the replay buffer.

<h3 id="data_spec"><code>data_spec</code></h3>

Returns the spec for items in the replay buffer.

<h3 id="device"><code>device</code></h3>

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="scope"><code>scope</code></h3>

<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.

<h3 id="table_fn"><code>table_fn</code></h3>

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).

## Methods

<h3 id="add_batch"><code>add_batch</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/replay_buffer.py">View
source</a>

``` python
add_batch(items)
```

Adds a batch of items to the replay buffer.

#### Args:

*   <b>`items`</b>: An item or list/tuple/nest of items to be added to the
    replay buffer. `items` must match the data_spec of this class, with a
    batch_size dimension added to the beginning of each tensor/array.

#### Returns:

Adds `items` to the replay buffer.

<h3 id="as_dataset"><code>as_dataset</code></h3>

``` python
as_dataset(
    *args,
    **kwargs
)
```

<h3 id="clear"><code>clear</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/replay_buffer.py">View
source</a>

``` python
clear()
```

Resets the contents of replay buffer.

#### Returns:

Clears the replay buffer contents.

<h3 id="gather_all"><code>gather_all</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/replay_buffer.py">View
source</a>

``` python
gather_all()
```

Returns all the items in buffer.

**NOTE** This method will soon be deprecated in favor of `as_dataset(...,
single_deterministic_pass=True)`.

#### Returns:

Returns all the items currently in the buffer. Returns a tensor
of shape [B, T, ...] where B = batch size, T = timesteps,
and the remaining shape is the shape spec of the items in the buffer.

<h3 id="get_next"><code>get_next</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/replay_buffer.py">View
source</a>

``` python
get_next(
    sample_batch_size=None,
    num_steps=None,
    time_stacked=True
)
```

Returns an item or batch of items from the buffer.

#### Args:

*   <b>`sample_batch_size`</b>: (Optional.) An optional batch_size to specify
    the number of items to return. If None (default), a single item is returned
    which matches the data_spec of this class (without a batch dimension).
    Otherwise, a batch of sample_batch_size items is returned, where each tensor
    in items will have its first dimension equal to sample_batch_size and the
    rest of the dimensions match the corresponding data_spec. See examples
    below.
*   <b>`num_steps`</b>: (Optional.) Optional way to specify that sub-episodes
    are desired. If None (default), in non-episodic replay buffers, a batch of
    single items is returned. In episodic buffers, full episodes are returned
    (note that sample_batch_size must be None in that case). Otherwise, a batch
    of sub-episodes is returned, where a sub-episode is a sequence of
    consecutive items in the replay_buffer. The returned tensors will have first
    dimension equal to sample_batch_size (if sample_batch_size is not None),
    subsequent dimension equal to num_steps, if time_stacked=True and remaining
    dimensions which match the data_spec of this class. See examples below.
*   <b>`time_stacked`</b>: (Optional.) Boolean, when true and num_steps > 1 it
    returns the items stacked on the time dimension. See examples below for
    details.

Examples of tensor shapes returned: (B = batch size, T = timestep, D = data
spec)

get_next(sample_batch_size=None, num_steps=None, time_stacked=True) return shape
(non-episodic): [D] return shape (episodic):
[T, D](T = full length of the episode) get_next(sample_batch_size=B,
num_steps=None, time_stacked=True) return shape (non-episodic): [B, D] return
shape (episodic): Not supported get_next(sample_batch_size=B, num_steps=T,
time_stacked=True) return shape: [B, T, D] get_next(sample_batch_size=None,
num_steps=T, time_stacked=False) return shape: ([D], [D], ..) T tensors in the
tuple get_next(sample_batch_size=B, num_steps=T, time_stacked=False) return
shape: ([B, D], [B, D], ..) T tensors in the tuple

#### Returns:

A 2-tuple containing:
  - An item or sequence of (optionally batched and stacked) items.
  - Auxiliary info for the items (i.e. ids, probs).

<h3 id="variables"><code>variables</code></h3>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/tf_uniform_replay_buffer.py">View
source</a>

``` python
variables()
```

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:

* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.
