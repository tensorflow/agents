<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.Checkpointer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize_or_restore"/>
<meta itemprop="property" content="save"/>
</div>

# tf_agents.utils.common.Checkpointer

## Class `Checkpointer`

Checkpoints training state, policy state, and replay_buffer state.





Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    ckpt_dir,
    max_to_keep=20,
    **kwargs
)
```

A class for making checkpoints.

If ckpt_dir doesn't exists it creates it.

#### Args:

* <b>`ckpt_dir`</b>: The directory to save checkpoints.
* <b>`max_to_keep`</b>: Maximum number of checkpoints to keep (if greater than the
    max are saved, the oldest checkpoints are deleted).
* <b>`**kwargs`</b>: Items to include in the checkpoint.



## Methods

<h3 id="initialize_or_restore"><code>initialize_or_restore</code></h3>

``` python
initialize_or_restore(session=None)
```

Initialize or restore graph (based on checkpoint if exists).

<h3 id="save"><code>save</code></h3>

``` python
save(global_step)
```

Save state to checkpoint.



