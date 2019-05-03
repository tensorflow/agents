<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.tensor_normalizer.EMATensorNormalizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="nested"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="normalize"/>
<meta itemprop="property" content="update"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.utils.tensor_normalizer.EMATensorNormalizer

## Class `EMATensorNormalizer`

TensorNormalizer with exponential moving avg. mean and var estimates.

Inherits From: [`TensorNormalizer`](../../../tf_agents/utils/tensor_normalizer/TensorNormalizer.md)



Defined in [`utils/tensor_normalizer.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/tensor_normalizer.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    tensor_spec,
    scope='normalize_tensor',
    norm_update_rate=0.001
)
```





## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="nested"><code>nested</code></h3>

True if tensor is nested, False otherwise.

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

Returns a tuple of tf variables owned by this EMATensorNormalizer.



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

<h3 id="copy"><code>copy</code></h3>

``` python
copy(scope=None)
```

Copy constructor for EMATensorNormalizer.

<h3 id="normalize"><code>normalize</code></h3>

``` python
normalize(
    tensor,
    clip_value=5.0,
    center_mean=True,
    variance_epsilon=0.001
)
```

Applies normalization to tensor.

#### Args:

* <b>`tensor`</b>: Tensor to normalize.
* <b>`clip_value`</b>: Clips normalized observations between +/- this value if
    clip_value > 0, otherwise does not apply clipping.
* <b>`center_mean`</b>: If true, subtracts off mean from normalized tensor.
* <b>`variance_epsilon`</b>: Epsilon to avoid division by zero in normalization.


#### Returns:

* <b>`normalized_tensor`</b>: Tensor after applying normalization.

<h3 id="update"><code>update</code></h3>

``` python
update(
    tensor,
    outer_dims=(0,)
)
```

Updates tensor normalizer variables.

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



