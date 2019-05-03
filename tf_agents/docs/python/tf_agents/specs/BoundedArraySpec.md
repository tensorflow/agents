<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.specs.BoundedArraySpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="maximum"/>
<meta itemprop="property" content="minimum"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="shape"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="check_array"/>
<meta itemprop="property" content="from_array"/>
<meta itemprop="property" content="from_spec"/>
</div>

# tf_agents.specs.BoundedArraySpec

## Class `BoundedArraySpec`

An `ArraySpec` that specifies minimum and maximum values.

Inherits From: [`ArraySpec`](../../tf_agents/specs/ArraySpec.md)

### Aliases:

* Class `tf_agents.specs.BoundedArraySpec`
* Class `tf_agents.specs.array_spec.BoundedArraySpec`



Defined in [`specs/array_spec.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/specs/array_spec.py).

<!-- Placeholder for "Used in" -->

Example usage:
```python
# Specifying the same minimum and maximum for every element.
spec = BoundedArraySpec((3, 4), np.float64, minimum=0.0, maximum=1.0)

# Specifying a different minimum and maximum for each element.
spec = BoundedArraySpec(
    (2,), np.float64, minimum=[0.1, 0.2], maximum=[0.9, 0.9])

# Specifying the same minimum and a different maximum for each element.
spec = BoundedArraySpec(
    (3,), np.float64, minimum=-10.0, maximum=[4.0, 5.0, 3.0])
```

Bounds are meant to be inclusive. This is especially important for
integer types. The following spec will be satisfied by arrays
with values in the set {0, 1, 2}:
```python
spec = BoundedArraySpec((3, 4), np.int, minimum=0, maximum=2)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    shape,
    dtype,
    minimum=None,
    maximum=None,
    name=None
)
```

Initializes a new `BoundedArraySpec`.

#### Args:

* <b>`shape`</b>: An iterable specifying the array shape.
* <b>`dtype`</b>: numpy dtype or string specifying the array dtype.
* <b>`minimum`</b>: Number or sequence specifying the maximum element bounds
    (inclusive). Must be broadcastable to `shape`.
* <b>`maximum`</b>: Number or sequence specifying the maximum element bounds
    (inclusive). Must be broadcastable to `shape`.
* <b>`name`</b>: Optional string containing a semantic name for the corresponding
    array. Defaults to `None`.


#### Raises:

* <b>`ValueError`</b>: If `minimum` or `maximum` are not broadcastable to `shape` or
    if the limits are outside of the range of the specified dtype.
* <b>`TypeError`</b>: If the shape is not an iterable or if the `dtype` is an invalid
    numpy dtype.



## Properties

<h3 id="dtype"><code>dtype</code></h3>

Returns a numpy dtype specifying the array dtype.

<h3 id="maximum"><code>maximum</code></h3>

Returns a NumPy array specifying the maximum bounds (inclusive).

<h3 id="minimum"><code>minimum</code></h3>

Returns a NumPy array specifying the minimum bounds (inclusive).

<h3 id="name"><code>name</code></h3>

Returns the name of the ArraySpec.

<h3 id="shape"><code>shape</code></h3>

Returns a `tuple` specifying the array shape.



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```



<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```



<h3 id="check_array"><code>check_array</code></h3>

``` python
check_array(array)
```

Return true if the given array conforms to the spec.

<h3 id="from_array"><code>from_array</code></h3>

``` python
from_array(
    array,
    name=None
)
```

Construct a spec from the given array or number.

<h3 id="from_spec"><code>from_spec</code></h3>

``` python
@classmethod
from_spec(
    cls,
    spec,
    name=None
)
```





