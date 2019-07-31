<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.get_contiguous_sub_episodes" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.get_contiguous_sub_episodes

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py">View
source</a>

Computes mask on sub-episodes which includes only contiguous components.

``` python
tf_agents.utils.common.get_contiguous_sub_episodes(next_time_steps_discount)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`next_time_steps_discount`</b>: Tensor of shape [batch_size, total_steps]
    corresponding to environment discounts on next time steps (i.e.
    next_time_steps.discount).

#### Returns:

A float Tensor of shape [batch_size, total_steps] specifying mask including only
contiguous components. Each row will be of the form [1.0] * a + [0.0] * b, where
a >= 1 and b >= 0, and in which the initial sequence of ones corresponds to a
contiguous sub-episode.
