<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.agents.dqn.dqn_agent.DqnLossInfo" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="td_loss"/>
<meta itemprop="property" content="td_error"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf_agents.agents.dqn.dqn_agent.DqnLossInfo

## Class `DqnLossInfo`

DqnLossInfo is stored in the `extras` field of the LossInfo instance.





Defined in [`agents/dqn/dqn_agent.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/dqn/dqn_agent.py).

<!-- Placeholder for "Used in" -->

Both `td_loss` and `td_error` have a validity mask applied to ensure that
no loss or error is calculated for episode boundaries.

td_loss: The **weighted** TD loss (depends on choice of loss metric and
  any weights passed to the DQN loss function.
td_error: The **unweighted** TD errors, which are just calculated as:

  ```
  td_error = td_targets - q_values
  ```

  These can be used to update Prioritized Replay Buffer priorities.

  Note that, unlike `td_loss`, `td_error` may contain a time dimension when
  training with RNN mode.  For `td_loss`, this axis is averaged out.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    td_loss,
    td_error
)
```

Create new instance of DqnLossInfo(td_loss, td_error)



## Properties

<h3 id="td_loss"><code>td_loss</code></h3>



<h3 id="td_error"><code>td_error</code></h3>





