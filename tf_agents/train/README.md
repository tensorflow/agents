# TF-Agents Actor-Learner API
The Actor-Learner API for TF-Agents provides an easy way to train Reinforcement
Learning models by separating data collection and model training logic into
modular components. The API is designed to be simple and modular by using easier
to understand Python training scripts, while handling the boilerplate and
complexity of Tensorflow. This new API also allows users to
easily scale their collection and training to run across multiple machines and
accelerators improving training speed (more on this in a later release).

Below is a brief description of the API and some code examples. We intend for
this API to be the default collection and training API going forward. **We
would love to hear your feedback on the API and its functionality, [please send
us your comments and suggestions using the tag #Actor-Learner](https://github.com/tensorflow/agents/issues)!**

### Replay Buffer
The Actor-Learner API uses [Reverb](https://github.com/deepmind/reverb)
as the replay buffer for storing experience replay. Before we look
at the Actor-Learner API, let us show an example of creating a Reverb
replay buffer.

Reverb replay buffer is contained in a separate server, that can be run alongside
the main training script or as a separate job. The server contains a table with
specified capacity and sampling and removing mechanisms (see
[Reverb replay buffer documentation]() for more details on those). Here's
a complete example of setting up a simple Reverb replay buffer with a uniform
sampling table of 1000 elements:

```python
  import reverb
  from tf_agents.agents.sac import sac_agent

  # initialize the agent
  agent = sac_agent.SacAgent(...)

  table = reverb.Table(
      table_name='experience',
      max_size=1000,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1))

  reverb_server = reverb.Server([table], port=8008)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name='uniform_table',
      local_server=reverb_server)
```

### Actor
The Actor class takes an instance of the environment
(as a `tf_agents.py_environment`), the policy and the replay buffer
[observer](https://github.com/tensorflow/agents/blob/fff4db7bc2bbad8ae12fbe92766cac38030474ad/tf_agents/replay_buffers/reverb_utils.py#L35) that adds trajectories to the buffer. Each Actor worker runs a sequence of
data collection steps given the values of the policy variables. The observed
experience is written into the replay buffer in each data collection step
using the observer. The Actor also logs summaries and metrics.

Below is an example of creating an Actor class:

```python
  from tf_agents.environments import suite_mujoco
  from tf_agents.replay_buffers import reverb_utils
  from tf_agents.train import actor
  from tf_agents.train.utils import train_utils

  REQUIRED_TIMESTEPS_PER_TRAIN_CALL=2

  collect_env = suite_mujoco.load('HalfCheetah-v2')
  collect_policy = agent.collect_policy
  train_step = train_utils.create_train_step()

  # reverb_replay buffer is created as shown above
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      py_client=reverb_replay.py_client,
      table_name='uniform_table',
      sequence_length=REQUIRED_TIMESTEPS_PER_TRAIN_CALL,
      stride_length=1)

  # create the Actor
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      summary_dir='/tmp/summaries',
      observers=[replay_buffer_observer])
```

To perform data collection, we can use the `run()` method of the Actor class
as follows:

```python
  for _ in range(10000):
    collect_actor.run()
```

### Learner
The Learner component performs gradient step updates to the policy variables
using experience data from the replay buffer. Its constructor takes the agent
and a function that generates the replay experience dataset. It also performs
summaries and checkpoints, storing the results in the pre-specified directory.
The Learner also takes an optional list of triggers, which is a list of callables
of the form `trigger(train_step)`. After every `run` call, every trigger is
called with the current `train_step` value. We provide a set of triggers, such
as `PolicySavedModelTrigger` and `StepPerSecondLogTrigger` but users can also
implement their own.

Below is an example of creating the experience dataset function from the replay
buffer and creating a Learner:

```python
  from tf_agents.train import learner
  from tf_agents.train import triggers

  # create the experience dataset function
  # reverb_replay buffer is created as in the example above
  dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size,
      num_steps=REQUIRED_TIMESTEPS_PER_TRAIN_CALL).prefetch(4)
  experience_dataset_fn = lambda: dataset

  # Add a simple trigger to log steps per second every 1000 steps
  learning_triggers = [
      triggers.StepPerSecondLogTrigger(train_step, interval=1000),
  ]

  # The learner writes summaries (at summary_interval) and checkpoints (at
  # checkpoint_interval) to the subdirectories under root_dir.
  # Users can add more summaries via the triggers.
  agent_learner = learner.Learner(
      root_dir='/tmp/train',
      train_step,
      agent,
      experience_dataset_fn,
      summary_interval=100,
      checkpoint_interval=100000,
      triggers=learning_triggers)
```

Now to run the train step, we can use the `run(iterations)` method of the
Learner. Each call to `run()` the Learner can run one or more training iterations:

```python
  for _ in range(num_iterations):
    collect_actor.run()
    agent_learner.run(iterations=1)
```
