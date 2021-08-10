# Distributed SAC Launch Instructions

## Launching locally:
In separate terminals, launch the following three (3) jobs. You can use a different
port number:

1) Reverb server:

```shell
  python tf_agents/experimental/distributed/examples/sac/sac_reverb_server.py -- \
  --root_dir=/tmp/sac_train/ \
  --port=8008 \
  --alsologtostderr
```

2) Collect job:

```shell
$  python tf_agents/experimental/distributed/examples/sac/sac_collect.py -- \
     --root_dir=/tmp/sac_train/ \
     --gin_bindings='collect.environment_name="HalfCheetah-v2"' \
     --replay_buffer_server_address=localhost:8008 \
     --variable_container_server_address=localhost:8008 \
     --alsologtostderr
```

3) Train job:

```shell
$  python tf_agents/experimental/distributed/examples/sac/sac_train.py -- \
     --root_dir=/tmp/sac_train/ \
     --gin_bindings='train.environment_name="HalfCheetah-v2"' \
     --gin_bindings='train.learning_rate=0.0003' \
     --replay_buffer_server_address=localhost:8008 \
     --variable_container_server_address=localhost:8008 \
     --alsologtostderr
```

4) Eval job (optional):

Not SAC specific. The evaluator job simply reads the greedy policy from an
arbitrary actor-learner `root_dir`, instantiates an environment (defined by the
GIN bindings to `evaluate.environment_name` and `evaluate.suite_load_fn`;
assumed the environment dependencies are already provided), then evaluates the
policy iteratively on policy parameters provided by the variable container.

```shell
$  python tf_agents/experimental/distributed/examples/ckpt_evaluator.py -- \
     --root_dir=/tmp/sac_train/ \
     --env_name='HalfCheetah-v2' \
     --alsologtostderr
```
