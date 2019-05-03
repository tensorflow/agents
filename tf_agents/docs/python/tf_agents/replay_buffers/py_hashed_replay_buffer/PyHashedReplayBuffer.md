<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.py_hashed_replay_buffer.PyHashedReplayBuffer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="capacity"/>
<meta itemprop="property" content="data_spec"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="add_batch"/>
<meta itemprop="property" content="as_dataset"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="gather_all"/>
<meta itemprop="property" content="get_next"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.replay_buffers.py_hashed_replay_buffer.PyHashedReplayBuffer

## Class `PyHashedReplayBuffer`

A Python-based replay buffer with optimized underlying storage.

Inherits From: [`PyUniformReplayBuffer`](../../../tf_agents/replay_buffers/py_uniform_replay_buffer/PyUniformReplayBuffer.md)



Defined in [`replay_buffers/py_hashed_replay_buffer.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/py_hashed_replay_buffer.py).

<!-- Placeholder for "Used in" -->

This replay buffer deduplicates data in the stored trajectories along the
last axis of the observation, which is useful, e.g., if you are performing
something like frame stacking. For example, if each observation is 4 stacked
84x84 grayscale images forming a shape [84, 84, 4], then the replay buffer
will separate out each of the images and depuplicate across each trajectory
in case an image is repeated.

Note: This replay buffer assumes that the items being stored are
trajectory.Trajectory instances.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    data_spec,
    capacity,
    log_interval=None
)
```





## Properties

<h3 id="capacity"><code>capacity</code></h3>

Returns the capacity of the replay buffer.

<h3 id="data_spec"><code>data_spec</code></h3>

Returns the spec for items in the replay buffer.

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="size"><code>size</code></h3>



<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> assert list(a.submodules) == [b, c]
>>> assert list(b.submodules) == [c]
>>> assert list(c.submodules) == []

#### Returns:

A sequence of all submodules.

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).

<h3 id="variables"><code>variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).



## Methods

<h3 id="__delattr__"><code>__delattr__</code></h3>

``` python
__delattr__(name)
```



<h3 id="__setattr__"><code>__setattr__</code></h3>

``` python
__setattr__(
    name,
    value
)
```

Support self.foo = trackable syntax.

<h3 id="add_batch"><code>add_batch</code></h3>

``` python
add_batch(items)
```

Adds a batch of items to the replay buffer.

#### Args:

* <b>`items`</b>: An item or list/tuple/nest of items to be added to the replay
    buffer. `items` must match the data_spec of this class, with a
    batch_size dimension added to the beginning of each tensor/array.

#### Returns:

Adds `items` to the replay buffer.

<h3 id="as_dataset"><code>as_dataset</code></h3>

``` python
as_dataset(
    sample_batch_size=None,
    num_steps=None,
    num_parallel_calls=None
)
```

Creates and returns a dataset that returns entries from the buffer.

A single entry from the dataset is equivalent to one output from
`get_next(sample_batch_size=sample_batch_size, num_steps=num_steps)`.

#### Args:

* <b>`sample_batch_size`</b>: (Optional.) An optional batch_size to specify the
    number of items to return. If None (default), a single item is returned
    which matches the data_spec of this class (without a batch dimension).
    Otherwise, a batch of sample_batch_size items is returned, where each
    tensor in items will have its first dimension equal to sample_batch_size
    and the rest of the dimensions match the corresponding data_spec.
* <b>`num_steps`</b>: (Optional.)  Optional way to specify that sub-episodes are
    desired. If None (default), a batch of single items is returned.
    Otherwise, a batch of sub-episodes is returned, where a sub-episode is a
    sequence of consecutive items in the replay_buffer. The returned tensors
    will have first dimension equal to sample_batch_size (if
    sample_batch_size is not None), subsequent dimension equal to num_steps,
    and remaining dimensions which match the data_spec of this class.
* <b>`num_parallel_calls`</b>: (Optional.) A `tf.int32` scalar `tf.Tensor`,
    representing the number elements to process in parallel. If not
    specified, elements will be processed sequentially.


#### Returns:

A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
  - An item or sequence of items or batch thereof
  - Auxiliary info for the items (i.e. ids, probs).

<h3 id="clear"><code>clear</code></h3>

``` python
clear()
```

Resets the contents of replay buffer.

#### Returns:

Clears the replay buffer contents.

<h3 id="gather_all"><code>gather_all</code></h3>

``` python
gather_all()
```

Returns all the items in buffer.

#### Returns:

Returns all the items currently in the buffer. Returns a tensor
of shape [B, T, ...] where B = batch size, T = timesteps,
and the remaining shape is the shape spec of the items in the buffer.

<h3 id="get_next"><code>get_next</code></h3>

``` python
get_next(
    sample_batch_size=None,
    num_steps=None,
    time_stacked=True
)
```

Returns an item or batch of items from the buffer.

#### Args:

* <b>`sample_batch_size`</b>: (Optional.) An optional batch_size to specify the
    number of items to return. If None (default), a single item is returned
    which matches the data_spec of this class (without a batch dimension).
    Otherwise, a batch of sample_batch_size items is returned, where each
    tensor in items will have its first dimension equal to sample_batch_size
    and the rest of the dimensions match the corresponding data_spec. See
    examples below.
* <b>`num_steps`</b>: (Optional.)  Optional way to specify that sub-episodes are
    desired. If None (default), in non-episodic replay buffers, a batch of
    single items is returned. In episodic buffers, full episodes are
    returned (note that sample_batch_size must be None in that case).
    Otherwise, a batch of sub-episodes is returned, where a sub-episode is a
    sequence of consecutive items in the replay_buffer. The returned tensors
    will have first dimension equal to sample_batch_size (if
    sample_batch_size is not None), subsequent dimension equal to num_steps,
    if time_stacked=True and remaining dimensions which match the data_spec
    of this class. See examples below.
* <b>`time_stacked`</b>: (Optional.) Boolean, when true and num_steps > 1 it returns
    the items stacked on the time dimension. See examples below for details.

  Examples of tensor shapes returned:
    (B = batch size, T = timestep, D = data spec)

    get_next(sample_batch_size=None, num_steps=None, time_stacked=True)
      return shape (non-episodic): [D]
      return shape (episodic): [T, D] (T = full length of the episode)
    get_next(sample_batch_size=B, num_steps=None, time_stacked=True)
      return shape (non-episodic): [B, D]
      return shape (episodic): Not supported
    get_next(sample_batch_size=B, num_steps=T, time_stacked=True)
      return shape: [B, T, D]
    get_next(sample_batch_size=None, num_steps=T, time_stacked=False)
      return shape: ([D], [D], ..) T tensors in the tuple
    get_next(sample_batch_size=B, num_steps=T, time_stacked=False)
      return shape: ([B, D], [B, D], ..) T tensors in the tuple

#### Returns:

A 2-tuple containing:
  - An item or sequence of (optionally batched and stacked) items.
  - Auxiliary info for the items (i.e. ids, probs).

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
...     return tf.matmul(x, self.w)

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

>>> mod = MyModule()
>>> mod(tf.ones([8, 32]))
<tf.Tensor: ...>
>>> mod.w
<tf.Variable ...'my_module/w:0'>

#### Args:

* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.



