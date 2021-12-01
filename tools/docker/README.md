# Docker build files for TF-Agents

This directory contains Docker build files used to run and build TF-Agents. Each
file has basic usage at the top. All of the commands below assume they are run
from the root of the github repository.

## Standard TF-Agents Docker

**Build with tf-agents nightly**

```shell
$ docker build -t tf_agents/core \
   --build-arg tf_agents_pip_spec=tf-agents-nightly[reverb] \
   -f tools/docker/ubuntu_tf_agents ./tools/docker
```

**Build with tf-agents latest stable**

```shell
$ docker build --pull -t tf_agents/core \
   --build-arg tf_agents_pip_spec=tf-agents[reverb] \
   --build-arg tensorflow_pip_spec=tensorflow \
   -f tools/docker/ubuntu_tf_agents ./tools/docker
```

**Run**

Starts the Docker image above and gives you a bash prompt and the TF-Agents'
repo mounted as `/workspace`.

```shell
$ sudo docker run -it -u $(id -u):$(id -g) -v $(pwd):/workspace \
    tf_agents/core bash
```

## TF-Agents plus Mujoco

The steps below build and run a docker with Mujoco.

Note: `tf_agents/core` is the base for this docker and built above.

```shell
$ docker build -t tf_agents/mujoco \
   -f tools/docker/ubuntu_mujoco ./tools/docker
```

Start the container:

```shell
$ docker run --rm -it \
  -v $(pwd):/workspace \
  tf_agents/mujoco bash
```

Start the container with GPU support:

```shell
$ docker run --rm -it \
 --gpus all \
  -v $(pwd):/workspace \
  tf_agents/mujoco bash
```
