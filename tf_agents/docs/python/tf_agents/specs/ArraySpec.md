<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.ArraySpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="shape"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="check_array"/>
<meta itemprop="property" content="from_array"/>
<meta itemprop="property" content="from_spec"/>
</div>

# tf_agents.specs.ArraySpec

## Class `ArraySpec`

Describes a numpy array or scalar shape and dtype.



### Aliases:

* Class `tf_agents.specs.ArraySpec`
* Class `tf_agents.specs.array_spec.ArraySpec`



Defined in [`specs/array_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/array_spec.py).

<!-- Placeholder for "Used in" -->

An `ArraySpec` allows an API to describe the arrays that it accepts or
returns, before that array exists.
The equivalent version describing a `tf.Tensor` is `TensorSpec`.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    shape,
    dtype,
    name=None
)
```

Initializes a new `ArraySpec`.

#### Args:

* <b>`shape`</b>: An iterable specifying the array shape.
* <b>`dtype`</b>: numpy dtype or string specifying the array dtype.
* <b>`name`</b>: Optional string containing a semantic name for the corresponding
    array. Defaults to `None`.


#### Raises:

* <b>`TypeError`</b>: If the shape is not an iterable or if the `dtype` is an invalid
    numpy dtype.



## Properties

<h3 id="dtype"><code>dtype</code></h3>

Returns a numpy dtype specifying the array dtype.

<h3 id="name"><code>name</code></h3>

Returns the name of the ArraySpec.

<h3 id="shape"><code>shape</code></h3>

Returns a `tuple` specifying the array shape.



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Checks if the shape and dtype of two specs are equal.

<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```



<h3 id="check_array"><code>check_array</code></h3>

``` python
check_array(array)
```

Return whether the given NumPy array conforms to the spec.

#### Args:

* <b>`array`</b>: A NumPy array or a scalar. Tuples and lists will not be converted
    to a NumPy array automatically; they will cause this function to return
    false, even if a conversion to a conforming array is trivial.


#### Returns:

True if the array conforms to the spec, False otherwise.

<h3 id="from_array"><code>from_array</code></h3>

``` python
@staticmethod
from_array(
    array,
    name=None
)
```

Construct a spec from the given array or number.

<h3 id="from_spec"><code>from_spec</code></h3>

``` python
@staticmethod
from_spec(spec)
```

Construct a spec from the given spec.



