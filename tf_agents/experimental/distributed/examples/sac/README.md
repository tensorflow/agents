# Distributed SAC Launch Instructions

## Launching locally:
In separate terminals, launch the following three (3) jobs. You can use a different
port number:

1) Reverb server:

```
  python tf_agents/experimental/distributed/examples/sac/sac_reverb_server.py -- \
  --root_dir=/tmp/sac_train/ \
  --port=8008 \
  --alsologtostderr
```

2) Collect job:

```
  python tf_agents/experimental/distributed/examples/sac/sac_collect.py -- \
  --root_dir=/tmp/sac_train/ \
  --replay_buffer_server_address=localhost:8008 \
  --variable_container_server_address=localhost:8008 \
  --alsologtostderr
```

3) Train job:

```
  python tf_agents/experimental/distributed/examples/sac/sac_train.py -- \
  --root_dir=/tmp/sac_train/ \
  --gin_bindings='train.learning_rate=0.0003' \
  --replay_buffer_server_address=localhost:8008 \
  --variable_container_server_address=localhost:8008 \
  --alsologtostderr
```
