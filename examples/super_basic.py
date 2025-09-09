import numpy as np
import pandas as pd
from bcsl.graph_utils import visualize_graph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from causal_pipe.causal_pipe import CausalPipeConfig
from causal_pipe.pipe_config import VariableTypes
from examples.utilities import compare_pipelines


def compare_super_basic_dataset(config: CausalPipeConfig):
    """
    Super-basic 2-node simulation: Var0 -> Var1
    Var0 ~ N(0,1)
    Var1 = 2*Var0 + eps, eps ~ N(0,1)
    """
    # Reproducibility
    np.random.seed(config.seed)
    config.study_name = "pipe_super_basic_dataset"

    # True graph
    true_graph = create_true_causal_graph_super_basic()
    visualize_graph(true_graph, title="True Causal Graph (SUPER BASIC: Var0 → Var1)", show=True)

    # Generate data
    n_samples = 500
    Var0 = np.random.normal(0, 1, n_samples)
    Var1 = 2 * Var0 + np.random.normal(0, 1, n_samples)

    data = pd.DataFrame({"Var0": Var0, "Var1": Var1})

    # Variable types
    config.variable_types = VariableTypes(continuous=["Var0", "Var1"])

    # Run your comparison
    compare_pipelines(data, config=config)


def create_true_causal_graph_super_basic() -> GeneralGraph:
    """
    Creates the true 2-node causal graph: Var0 -> Var1.
    """
    node_names = ["Var0", "Var1"]
    nodes = {name: GraphNode(name) for name in node_names}
    graph = GeneralGraph(list(nodes.values()))

    edge = Edge(
        nodes["Var0"],
        nodes["Var1"],
        Endpoint.TAIL,   # source
        Endpoint.ARROW,  # target
    )
    graph.add_edge(edge)

    return graph
