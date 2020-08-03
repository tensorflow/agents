# Docker build files for TF-Agents

This directory contains Docker build files used to run and build TF-Agents. Each
file should have basic usage at the top. Below is a primer and some extra hints
and snippets.

## Building and running

Check the usage section contained in the Docker build files for specifics. These
instructions are general to any Docker.

**Build**

The command below builds a Docker image with the most recent stable version of
TF-Agents and tags it as `tf_agents/release_test`.

```bash
docker build -t tf_agents/release_test -f ubuntu_1804_tf_agents .
```

**Run**

Starts the Docker image above and gives you a bash prompt and the TF-Agents'
repo mounted as `/workspace`.

```bash
# cd to the github root
sudo docker run -it -u $(id -u):$(id -g) -v $(pwd):/workspace \
  tf_agents/release_test bash
```

## Extras

If you want to use Mujoco, the snippet below can be added to the Docker build
file near the end. You do need to include your license key during the build
or at runtime.

```bash
######################################################
# Mujoco install. Only the install no license.
######################################################
RUN apt-get update && apt-get install -y \
  --no-install-recommends \
  libglew-dev \
  libosmesa6-dev \
  patchelf

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

# Mujoco will not install without a key. This is a fake key.
RUN echo 'This is a fake key, you will need to put your key here.' > /root/.mujoco/mjkey.txt

RUN pip install -U 'mujoco-py<2.1,>=2.0'
```
