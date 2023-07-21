# Actor-Learner API for Distributed Collection and Training

In this README we describe the use of the TF-Agents Actor-Learner API for
distributed collection and training. Please read the
[API overview](https://github.com/tensorflow/agents/tree/master/tf_agents/train/README.md)
first.

The TF-Agents Actor-Learner API supports both distributed data collection across
machines and distributed training across workers and accelerators.

In distributed collection, we can run multiple (10s-1000s) Actor workers each
collecting experience data given a policy. In distributed training, in the
Learner we use data-parallelism to split a batch of training data across
multiple accelerator devices, such as GPUs or TPU cores, one one machine (and
in the future across multiple workers as well). Using
both parallel training and data collection can significantly improve time to
convergence when training RL models.


## Distributed Architecture

The following diagram gives an overview of the distributed architecture using
the Actor-Learner API:

![Actor-Learner Distributed Architecture](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/actor_learner_distributed_architecture.png)

The main components in the distributed collection and training setup are the same
as for single-machine setup, except for an additional Variable Container component:
- a set of Actors that perform data collection
- a Learner that performs the training updates
- a Reverb replay buffer that Actors and the Learner use to communicate experience
data through
- (new) a Reverb variable container that Actors and Learner use to communicate policy
weights and other variables through

Note that due to the modular architecture, the replay buffer and
variable container servers can be co-located with the Learner or reside
on separate machines.

#### Variable Container and Reverb

An additional component that we need to run distributed collection is a Variable
Container. Variable container stores variables such as policy weights or
`train_step` that are shared between the Actors and the Learner. During training,
the Learner can push the new variable values to
the container and the Actor can request the latest values before performing data
collection.

Similar to the Replay Buffer, we use [Reverb](https://github.com/deepmind/reverb)
to implement the variable container. Below is an example of creating a variable
container, both the Actor and Learner use the same code to create the container.
Note that this requires an instance of a reverb server, you can take a look at
how a local server (that runs in the same binary) is constructed in the [train/README.md](https://github.com/tensorflow/agents/blob/master/tf_agents/train/README.md)

For distributed setup, we can run a separate instance of the Reverb server
serving both the replay buffer and variable container tables. Here is an example
of setting up a separate reverb server:

```python
  port = 8008
  min_table_size_before_sampling = 1
  replay_buffer_capacity = 1000
  # A very simple rate limiter, blocks until there is at least
  # `min_table_size_before_sampling` elements in the tables
  experience_rate_limiter = reverb.rate_limiters.MinSize(
      min_table_size_before_sampling)

  # get collect data spec from the collect policy for replay buffer table
  replay_buffer_signature = tensor_spec.from_spec(
      collect_policy.collect_data_spec)

  train_step = train_utils.create_train_step()
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  # variables signature for variable container table
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
      variables)

  # create the server
  server = reverb.Server(
      tables=[
          reverb.Table(  # Replay buffer storing experience.
              name=reverb_replay_buffer.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=experience_rate_limiter,
              max_size=replay_buffer_capacity,
              max_times_sampled=0,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=1,
              max_times_sampled=0,
              signature=variable_container_signature,
          ),
      ],
      port=port)
```

And now we can create a variable container that connects to the reverb server
via the `variable_container_server_address` (and similarly to the replay buffer):

```python
  from tf_agents.experimental.distributed import reverb_variable_container

  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
```

Then the Learner worker can push variables to the container with:

```python
  variable_container.push(variables)
```

And the Actor worker can (in-place) update the variables dictionary with the
values stored in the variable container:

```python
  variable_container.update(variables)
```

## Distributed Data Collection

To distribute data collection, we run multiple instances of the Actor workers,
each of which runs the following logic:

```python
  from tf_agents.replay_buffers import reverb_utils

  num_iterations = 10000

  replay_buffer_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb.Client(replay_buffer_server_address),
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      sequence_length=2,
      stride_length=1)

  # As in the example above, create the Actor, pass it the reverb replay buffer
  # observer to write experience data to the buffer.
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      summary_dir='/tmp/summaries',
      observers=[replay_buffer_observer])

  # Run num_iterations iterations, performing data collection and updating the
  # policy variables after each collection call. Note that the Actor performs
  # steps_per_run collection steps per run call.
  for _ in range(num_iterations):
    logging.info('Collecting with policy at step: %d', train_step.numpy())
    collect_actor.run()
    variable_container.update(variables)

```

Each worker runs `num_iterations` of collection rollouts. During each collection
step, the Actor sends the observed result to the passed `replay_buffer_observer`.
In the distributed collection case, we use the `ReverbAddTrajectoryObserver`
that adds the trajectory to the Reverb replay buffer. The Learner then can
sample the trajectories from the replay buffer during training.

## Distributed Training

We use the [DistributionStrategy API](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)
in the TF-Agents Learner to enable running
the train step computation across multiple devices (called replicas) such as
multiple GPUs or TPU cores. We parallelize the train step using data-parallelism,
i.e. each of the N cores on the Learner worker gets a batch of size B/N where
B is the total batch size read per step.

The diagram below shows an example of the Learner
running the train step update on 4 GPUs. The train step receives a batch of
training data, splits it across the GPUs, computes the forward step, aggregates
and computes the MEAN of the loss, computes the backward step and performs a
gradient variable update. This is all done transparently to the user.

![Distributed Learner on 4 GPUs](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/learner_detail.png)

With the DistributionStrategy API it is quite easy to switch between running
the train step on GPUs (using MirroredStrategy) to TPUs
(using TPUStrategy) without changing any of the training logic.
Below is an example of using the Learner with `MirroredStrategy` to parallelize
training across a GPU on one machine:

```python
  num_iterations = 1000000
  strategy = tf.distribute.MirroredStrategy()

  # Define an experience dataset function that is passed to each of the distributed
  # replicas. This is used to read training data in parallel for the replicas.
  def experience_dataset_fn():
    with strategy.scope():
      return replay_buffer.as_dataset(
          sample_batch_size=batch_size, num_steps=2).prefetch(3)

  # Create the learner.
  sac_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
      strategy=strategy)

  # Run num_iterations of training, push variable update after each run call.
  # Note that each run call can run learner_iterations_per_call iterations. This
  # is especially important for more efficient TPU utilization. This number needs
  # to be sufficiently high to get the benefits from tf.function optimizations
  # on TPU (to save on launch overhead), but not too high to be able to perform
  # actions between those calls (i.e. checkpointing) with a reasonable frequency.
  learner_iterations_per_call = 100

  for _ in range(num_iterations):
    sac_learner.run(iterations=learner_iterations_per_call)
    variable_container.push(variables)
```
