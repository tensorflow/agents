<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.replay_buffers.py_hashed_replay_buffer.FrameBuffer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="add_frame"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="compress"/>
<meta itemprop="property" content="decompress"/>
<meta itemprop="property" content="deserialize"/>
<meta itemprop="property" content="on_delete"/>
<meta itemprop="property" content="serialize"/>
</div>

# tf_agents.replay_buffers.py_hashed_replay_buffer.FrameBuffer

## Class `FrameBuffer`

Saves some frames in a memory efficient way.





Defined in [`replay_buffers/py_hashed_replay_buffer.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/replay_buffers/py_hashed_replay_buffer.py).

<!-- Placeholder for "Used in" -->

Thread safety: cannot add multiple frames in parallel.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__()
```





## Methods

<h3 id="__len__"><code>__len__</code></h3>

``` python
__len__()
```



<h3 id="add_frame"><code>add_frame</code></h3>

``` python
add_frame(frame)
```

Add a frame to the buffer.

#### Args:

* <b>`frame`</b>: Numpy array.


#### Returns:

A deduplicated frame.

<h3 id="clear"><code>clear</code></h3>

``` python
clear()
```



<h3 id="compress"><code>compress</code></h3>

``` python
compress(
    observation,
    split_axis=-1
)
```



<h3 id="decompress"><code>decompress</code></h3>

``` python
decompress(
    observation,
    split_axis=-1
)
```



<h3 id="deserialize"><code>deserialize</code></h3>

``` python
deserialize(string_value)
```

Callback for `PythonStateWrapper` to deserialize the array.

<h3 id="on_delete"><code>on_delete</code></h3>

``` python
on_delete(
    observation,
    split_axis=-1
)
```



<h3 id="serialize"><code>serialize</code></h3>

``` python
serialize()
```

Callback for `PythonStateWrapper` to serialize the dictionary.



