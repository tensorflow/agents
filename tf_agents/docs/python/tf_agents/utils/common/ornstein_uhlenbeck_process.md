<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_agents.utils.common.ornstein_uhlenbeck_process" />
<meta itemprop="path" content="Stable" />
</div>

# tf_agents.utils.common.ornstein_uhlenbeck_process

An op for generating noise from a zero-mean Ornstein-Uhlenbeck process.

``` python
tf_agents.utils.common.ornstein_uhlenbeck_process(
    initial_value,
    damping=0.15,
    stddev=0.2,
    seed=None,
    scope='ornstein_uhlenbeck_noise'
)
```



Defined in [`utils/common.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/utils/common.py).

<!-- Placeholder for "Used in" -->

The Ornstein-Uhlenbeck process is a process that generates temporally
correlated noise via a random walk with damping. This process describes
the velocity of a particle undergoing brownian motion in the presence of
friction. This can be useful for exploration in continuous action environments
with momentum.

The temporal update equation is:
`x_next = (1 - damping) * x + N(0, std_dev)`

#### Args:

* <b>`initial_value`</b>: Initial value of the process.
* <b>`damping`</b>: The rate at which the noise trajectory is damped towards the mean.
    We must have 0 <= damping <= 1, where a value of 0 gives an undamped
    random walk and a value of 1 gives uncorrelated Gaussian noise. Hence
    in most applications a small non-zero value is appropriate.
* <b>`stddev`</b>: Standard deviation of the Gaussian component.
* <b>`seed`</b>: Seed for random number generation.
* <b>`scope`</b>: Scope of the variables.


#### Returns:

An op that generates noise.