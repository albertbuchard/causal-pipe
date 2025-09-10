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


def compare_non_linear_dataset(config: CausalPipeConfig):
    """Generate and compare a simple non-linear dataset."""
    np.random.seed(config.seed)
    config.study_name = "pipe_non_linear_dataset"

    true_graph = create_true_causal_graph_non_linear()
    visualize_graph(true_graph, title="True Causal Graph (NON_LINEAR)", show=True)

    n_samples = 500

    Var0 = np.random.normal(0, 1, n_samples)
    Var1 = np.random.normal(0, 1, n_samples)
    Var2 = Var0**2 + np.random.normal(0, 0.5, n_samples)
    Var3 = np.sin(Var1) + Var2 + np.random.normal(0, 0.5, n_samples)

    data = pd.DataFrame({"Var0": Var0, "Var1": Var1, "Var2": Var2, "Var3": Var3})

    config.variable_types = VariableTypes(continuous=["Var0", "Var1", "Var2", "Var3"])

    compare_pipelines(data, config=config)


def create_true_causal_graph_non_linear() -> GeneralGraph:
    """Creates the true causal graph for the non-linear dataset."""
    node_names = ["Var0", "Var1", "Var2", "Var3"]
    nodes = {name: GraphNode(name) for name in node_names}
    graph = GeneralGraph(list(nodes.values()))

    true_edges = [("Var0", "Var2"), ("Var1", "Var3"), ("Var2", "Var3")]

    for source, target in true_edges:
        edge = Edge(nodes[source], nodes[target], Endpoint.TAIL, Endpoint.ARROW)
        graph.add_edge(edge)

    return graph
