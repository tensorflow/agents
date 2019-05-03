<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.numpy_storage.NumpyStorage" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="set"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.utils.numpy_storage.NumpyStorage

## Class `NumpyStorage`

A class to store nested objects in a collection of numpy arrays.





Defined in [`utils/numpy_storage.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/numpy_storage.py).

<!-- Placeholder for "Used in" -->

If a data_spec of `{'foo': ArraySpec(shape=(4,), dtype=np.uint8), 'bar':
ArraySpec(shape=(3, 7), dtype=np.float32)}` were used, then this would create
two arrays, one for the 'foo' key and one for the 'bar' key. The .get and
.set methods would return/take Python dictionaries, but break down the
component arrays before storing them.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    data_spec,
    capacity
)
```

Creates a NumpyStorage object.

#### Args:

* <b>`data_spec`</b>: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
    single item that can be stored in this table.
* <b>`capacity`</b>: The maximum number of items that can be stored in the buffer.


#### Raises:

* <b>`ValueError`</b>: If data_spec is not an instance or nest of ArraySpecs.



## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

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

<h3 id="get"><code>get</code></h3>

``` python
get(idx)
```

Get value stored at idx.

<h3 id="set"><code>set</code></h3>

``` python
set(
    table_idx,
    value
)
```

Set table_idx to value.

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



