<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.metrics.py_metric.PyStepMetric" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="prefix"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="summary_op"/>
<meta itemprop="property" content="summary_placeholder"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__delattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="aggregate"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="log"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="tf_summaries"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tf_agents.metrics.py_metric.PyStepMetric

## Class `PyStepMetric`

Defines the interface for metrics that operate on trajectories.

Inherits From: [`PyMetric`](../../../tf_agents/metrics/py_metric/PyMetric.md)



Defined in [`metrics/py_metric.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metric.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    name,
    prefix='Metrics'
)
```

Creates a metric.



## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="prefix"><code>prefix</code></h3>

Prefix for the metric.

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

<h3 id="summary_op"><code>summary_op</code></h3>

TF summary op for this metric.

<h3 id="summary_placeholder"><code>summary_placeholder</code></h3>

TF placeholder to be used for the result of this metric.

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

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(*args)
```

Method to update the metric contents.

To change the behavior of this function, override the call method.

Different subclasses might use this differently. For instance, the
PyStepMetric takes in a trajectory, while the CounterMetric takes no
parameters.

#### Args:

* <b>`*args`</b>: See call method of subclass for specific arguments.

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

<h3 id="aggregate"><code>aggregate</code></h3>

``` python
aggregate(metrics)
```

Aggregates a list of metrics.

The default behaviour is to return the average of the metrics.

#### Args:

* <b>`metrics`</b>: a list of metrics, of the same class.

#### Returns:

The result of aggregating this metric.

<h3 id="call"><code>call</code></h3>

``` python
call(trajectory)
```

Processes a trajectory to update the metric.

#### Args:

* <b>`trajectory`</b>: A trajectory.Trajectory.

<h3 id="log"><code>log</code></h3>

``` python
log()
```



<h3 id="reset"><code>reset</code></h3>

``` python
reset()
```

Resets internal stat gathering variables used to compute the metric.

<h3 id="result"><code>result</code></h3>

``` python
result()
```

Evaluates the current value of the metric.

<h3 id="tf_summaries"><code>tf_summaries</code></h3>

``` python
tf_summaries(
    train_step=None,
    step_metrics=()
)
```

Build TF summary op and placeholder for this metric.

To execute the op, call py_metric.run_summaries.

#### Args:

* <b>`train_step`</b>: Step counter for training iterations. If None, no metric is
    generated against the global step.
* <b>`step_metrics`</b>: Step values to plot as X axis in addition to global_step.


#### Returns:

The summary op.


#### Raises:

* <b>`RuntimeError`</b>: If this method has already been called (it can only be
    called once).
* <b>`ValueError`</b>: If any item in step_metrics is not of type PyMetric or
    tf_metric.TFStepMetric.

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



