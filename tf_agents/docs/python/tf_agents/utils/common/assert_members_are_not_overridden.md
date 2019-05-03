<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.assert_members_are_not_overridden" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.assert_members_are_not_overridden

Asserts public members of `base_cls` are not overridden in `instance`.

``` python
tf_agents.utils.common.assert_members_are_not_overridden(
    base_cls,
    instance,
    white_list=(),
    black_list=()
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

If both `white_list` and `black_list` are empty, no public member of
`base_cls` can be overridden. If a `white_list` is provided, only public
members in `white_list` can be overridden. If a `black_list` is provided,
all public members except those in `black_list` can be overridden. Both
`white_list` and `black_list` cannot be provided at the same, if so a
ValueError will be raised.

#### Args:

* <b>`base_cls`</b>: A Base class.
* <b>`instance`</b>: An instance of a subclass of `base_cls`.
* <b>`white_list`</b>: Optional list of `base_cls` members that can be overridden.
* <b>`black_list`</b>: Optional list of `base_cls` members that cannot be overridden.


#### Raises:

ValueError if both white_list and black_list are provided.