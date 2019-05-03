<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.table.Table" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="slots"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="read"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="with_name_scope"/>
<meta itemprop="property" content="write"/>
</div>

# tf_agents.replay_buffers.table.Table

## Class `Table`

A table that can store Tensors or nested Tensors.





Defined in [`replay_buffers/table.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/table.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    tensor_spec,
    capacity,
    scope='Table'
)
```

Creates a table.

#### Args:

* <b>`tensor_spec`</b>: A nest of TensorSpec representing each value that can be
    stored in the table.
* <b>`capacity`</b>: Maximum number of values the table can store.
* <b>`scope`</b>: Variable scope for the Table.

#### Raises:

* <b>`ValueError`</b>: If the names in tensor_spec are empty or not unique.



## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="slots"><code>slots</code></h3>



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

<h3 id="read"><code>read</code></h3>

``` python
read(
    rows,
    slots=None
)
```

Returns values for the given rows.

#### Args:

* <b>`rows`</b>: A scalar/list/tensor of location(s) to read values from. If rows is
    a scalar, a single value is returned without a batch dimension. If rows
    is a list of integers or a rank-1 int Tensor a batch of values will be
    returned with each Tensor having an extra first dimension equal to the
    length of rows.
* <b>`slots`</b>: Optional list/tuple/nest of slots to read from. If None, all
    tensors at the given rows are retrieved and the return value has the
    same structure as the tensor_spec. Otherwise, only tensors with names
    matching the slots are retrieved, and the return value has the same
    structure as slots.


#### Returns:

Values at given rows.

<h3 id="variables"><code>variables</code></h3>

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

<h3 id="write"><code>write</code></h3>

``` python
write(
    rows,
    values,
    slots=None
)
```

Returns ops for writing values at the given rows.

#### Args:

* <b>`rows`</b>: A scalar/list/tensor of location(s) to write values at.
* <b>`values`</b>: A nest of Tensors to write. If rows has more than one element,
    values can have an extra first dimension representing the batch size.
    Values must have the same structure as the tensor_spec of this class
    if `slots` is None, otherwise it must have the same structure as
    `slots`.
* <b>`slots`</b>: Optional list/tuple/nest of slots to write. If None, all tensors
    in the table are updated. Otherwise, only tensors with names matching
    the slots are updated.


#### Returns:

Ops for writing values at rows.



