import os
import sys

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None


def check_pydot():
    """Returns True if PyDot is available."""
    return pydot is not None


def check_graphviz():
    """Returns True if both PyDot and Graphviz are available."""
    if not check_pydot():
        return False
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, pydot.InvocationException):
        return False


def add_edge(dot, src, dst):
    if not dot.get_edge(src, dst):
        dot.add_edge(pydot.Edge(src, dst))


def add_edge_node(dot, node, next_node):
    if node['type'] == "sequential":
        for i in range(len(node['nodes']) - 1):
            add_edge_node(dot, node['nodes'][i], node['nodes'][i + 1])
        add_edge_node(dot, node['nodes'][-1], next_node)
    elif node['type'] == "nest":
        for i in range(len(node['nodes'])):
            add_edge_node(dot, node['nodes'][i], next_node)
    elif next_node['type'] == "sequential":
        for i in range(len(next_node['nodes']) - 1):
            add_edge_node(dot, node, next_node['nodes'][i])
        add_edge_node(dot, node, next_node['nodes'][0])
    elif next_node['type'] == "nest":
        for i in range(len(next_node['nodes'])):
            add_edge_node(dot, node, next_node['nodes'][i])
    else:
        add_edge(dot, node['id'], next_node['id'])


def make_node(id):
    return {"id": id, "type": "node"}


def model_to_dot(
    model,
    subgraph=False,
    dpi=96,
    depth=4,
):

    if not model.built:
        raise ValueError(
            "This model has not yet been built. "
            "Build the model first by calling `build()` or by calling "
            "the model on a batch of data."
        )

    from tf_agents.networks import NestMap
    from tf_agents.networks import NestFlatten
    from tf_agents.networks import sequential

    if not check_pydot():
        raise ImportError(
            "You must install pydot (`pip install pydot`) for "
            "model_to_dot to work."
        )

    if subgraph:
        dot = pydot.Cluster(style="dashed", graph_name=model.name)
        dot.set("label", model.name)
        dot.set("labeljust", "l")
    else:
        dot = pydot.Dot()
        dot.set("rankdir", "TB")
        dot.set("dpi", dpi)
        dot.set_node_defaults(shape="record")

    layers = model.layers
    listIdNode = {"nodes": [], "type": (
        "nest" if isinstance(model, NestMap) else "sequential")}

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__

        # Create node's label.
        label = "{0}|{1}".format(class_name, layer_name)

        def format_shape(shape):
            return (
                str(shape)
                .replace(str(None), "None")
                .replace("{", r"\{")
                .replace("}", r"\}")
            )

        try:
            outputlabels = format_shape(layer.output_shape)
        except AttributeError:
            outputlabels = "?"
        if hasattr(layer, "input_shape"):
            inputlabels = format_shape(layer.input_shape)
        elif hasattr(layer, "input_shapes"):
            inputlabels = ", ".join(
                [format_shape(ishape) for ishape in layer.input_shapes]
            )
        else:
            inputlabels = "?"
        label = "{%s}|{input:|output:}|{{%s}|{%s}}" % (
            label,
            inputlabels,
            outputlabels,
        )

        if depth == 0:
            listIdNode['nodes'].append(make_node(layer_id))
            node = pydot.Node(layer_id, label=label)
            dot.add_node(node)
            continue

        if isinstance(layer, sequential.Sequential) or isinstance(layer, NestMap):
            submodel_wrapper, sub_listIdNode = model_to_dot(
                layer, subgraph=True, dpi=dpi, depth=depth-1)
            listIdNode['nodes'].append(sub_listIdNode)
            dot.add_subgraph(submodel_wrapper)
        else:
            listIdNode['nodes'].append(make_node(layer_id))
            node = pydot.Node(layer_id, label=label)
            dot.add_node(node)

    # Add edges between nodes.
    if not subgraph and isinstance(model, sequential.Sequential):
        for i in range(len(listIdNode['nodes']) - 1):
            node = listIdNode['nodes'][i]
            next_node = listIdNode['nodes'][i + 1]

            add_edge_node(dot, node, next_node)

    return dot, listIdNode


def print_msg(message, line_break=True):
    if line_break:
        sys.stdout.write(message + "\n")
    else:
        sys.stdout.write(message)
    sys.stdout.flush()


def path_to_string(path):
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def plot_model(
    model,
    to_file="model.png",
    subgraph=False,
    dpi=96,
    depth=4,
):

    if not model.built:
        raise ValueError(
            "This model has not yet been built. "
            "Build the model first by calling `build()` or by calling "
            "the model on a batch of data."
        )

    if not check_graphviz():
        message = (
            "You must install pydot (`pip install pydot`) "
            "and install graphviz "
            "(see instructions at https://graphviz.gitlab.io/download/) "
            "for plot_model to work."
        )
        if "IPython.core.magics.namespace" in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            print_msg(message)
            return
        else:
            raise ImportError(message)

    dot, _ = model_to_dot(
        model,
        subgraph=subgraph,
        dpi=dpi,
        depth=depth,
    )
    to_file = path_to_string(to_file)
    if dot is None:
        return
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = "png"
    else:
        extension = extension[1:]
    # Save image to disk.
    dot.write(to_file, format=extension)
    # Return the image as a Jupyter Image object, to be displayed in-line.
    # Note that we cannot easily detect whether the code is running in a
    # notebook, and thus we always return the Image if Jupyter is available.
    if extension != "pdf":
        try:
            from IPython import display

            return display.Image(filename=to_file)
        except ImportError:
            pass
