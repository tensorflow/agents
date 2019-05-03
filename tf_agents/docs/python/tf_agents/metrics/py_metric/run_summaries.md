<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.metrics.py_metric.run_summaries" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.metrics.py_metric.run_summaries

Execute summary ops for py_metrics.

``` python
tf_agents.metrics.py_metric.run_summaries(
    metrics,
    session=None
)
```



Defined in [`metrics/py_metric.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/metrics/py_metric.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`metrics`</b>: A list of py_metric.Base objects.
* <b>`session`</b>: A TensorFlow session-like object. If it is not provided, it will
    use the current TensorFlow session context manager.


#### Raises:

* <b>`RuntimeError`</b>: If .tf_summaries() was not previously called on any of the
    `metrics`.
* <b>`AttributeError`</b>: If session is not provided and there is no default session
    provided by a context manager.