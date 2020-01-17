load("//research/colab:build_defs.bzl", "colab_binary", "colab_notebook_test")

package(
    default_visibility = [
        "//third_party/py/tf_agents:default_visibility",
    ],
)

licenses(["notice"])  # Apache License 2.0

colab_binary(
    name = "colab",
    kernel_init = [
        # Set up display.
        "colab_kernel_init.py",
    ],
    # We explicitly opt into hermetic Python (colab_binary is non-hermetic by default) to resolve
    # a problem relating to tcmalloc and the OSMesa rendering backend for MuJoCo (b/32466075).
    launcher = "//devtools/python/launcher",
    python_version = "PY3",
    deps = [
        "//file/liball",
        "//learning/brain/python/client:notebook_deps",
        "//third_party/py/OpenGL",
        "//third_party/py/gfootball",
        "//third_party/py/imageio",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/tf_agents/agents",
        "//third_party/py/tf_agents/bandits/agents",
        "//third_party/py/tf_agents/bandits/environments",
        "//third_party/py/tf_agents/bandits/metrics",
        "//third_party/py/tf_agents/bandits/policies",
        "//third_party/py/tf_agents/drivers",
        "//third_party/py/tf_agents/environments",
        "//third_party/py/tf_agents/environments:suites",
        "//third_party/py/tf_agents/eval",
        "//third_party/py/tf_agents/metrics",
        "//third_party/py/tf_agents/networks",
        "//third_party/py/tf_agents/policies",
        "//third_party/py/tf_agents/replay_buffers",
        "//third_party/py/tf_agents/specs",
        "//third_party/py/tf_agents/trajectories",
        "//third_party/py/tf_agents/utils:common",
        "//third_party/py/tf_agents/utils:example_encoding",
        "//third_party/py/tf_agents/utils:nest_utils",
        "//third_party/py/tree",
    ],
)

colab_notebook_test(
    name = "environments_colab_test",
    colab_binary = ":colab",
    default_cell_diff = "ignore",
    ipynb = "2_environments_tutorial.ipynb",
)

colab_notebook_test(
    name = "policies_colab_test",
    colab_binary = ":colab",
    default_cell_diff = "ignore",
    ipynb = "3_policies_tutorial.ipynb",
    tags = [
        "tf_v1_only",  # TODO(b/123882307)
    ],
)

colab_notebook_test(
    name = "drivers_tutorial_test",
    colab_binary = ":colab",
    default_cell_diff = "ignore",
    ipynb = "4_drivers_tutorial.ipynb",
)

colab_notebook_test(
    name = "replay_buffers_tutorial_test",
    colab_binary = ":colab",
    default_cell_diff = "ignore",
    ipynb = "5_replay_buffers_tutorial.ipynb",
)

colab_notebook_test(
    name = "checkpointer_policysaver_tutorial_test",
    colab_binary = ":colab",
    default_cell_diff = "ignore",
    ipynb = "10_checkpointer_policysaver_tutorial.ipynb",
)
