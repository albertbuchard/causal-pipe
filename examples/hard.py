import numpy as np
import pandas as pd
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from bcsl.graph_utils import visualize_graph

from causal_pipe.causal_pipe import CausalPipeConfig
from causal_pipe.pipe_config import VariableTypes
from examples.utilities import compare_pipelines


def compare_hard_dataset(config: CausalPipeConfig):
    # Generate synthetic data for testing
    np.random.seed(config.seed)

    n_samples = 1000  # Increased sample size for complexity

    # Define True Causal Graph with a latent confounder U affecting Var0 and Var1
    # Note: U is latent and not included in the observed data
    true_graph = create_true_causal_graph_hard()
    visualize_graph(true_graph, title="True Causal Graph (HARD)", show=True)

    # Latent confounder (not included in the dataset directly)
    # Assume there's a latent variable U that affects Var0 and Var1
    U = np.random.normal(0, 1, n_samples)

    # Add multiplicative noise to Var3, Var5, and Var7
    noise_factor_var3 = 1 + np.random.normal(
        0, 0.1, n_samples
    )  # 10% multiplicative noise
    noise_factor_var5 = 1 + np.random.normal(0, 0.1, n_samples)
    noise_factor_var7 = 1 + np.random.normal(0, 0.1, n_samples)

    # Observed variables
    Var0 = U + np.random.normal(0, 1, n_samples)  # Var0 influenced by U
    Var1 = U + np.random.normal(0, 1, n_samples)  # Var1 influenced by U
    Var2 = Var0 * 2 + np.random.normal(0, 1, n_samples)  # Var2 depends on Var0
    Var3 = (
        Var1 * -1.5 + Var2 * 0.5 + np.random.normal(0, 1, n_samples)
    ) * noise_factor_var3  # Var3 depends on Var1 and Var2
    Var4 = Var2 * 1.2 + np.random.normal(0, 1, n_samples)  # Var4 depends on Var2
    Var5 = (
        Var3 + Var4 + np.random.normal(0, 1, n_samples)
    ) * noise_factor_var5  # Var5 depends on Var3 and Var4
    Var6 = Var5 * 0.7 + np.random.normal(0, 1, n_samples)  # Var6 depends on Var5
    Var7 = (
        Var1 * 0.3 + Var6 * 1.5 + np.random.normal(0, 1, n_samples)
    ) * noise_factor_var7  # Var7 depends on Var1 and Var6
    Var8 = Var4 * -0.8 + np.random.normal(0, 1, n_samples)  # Var8 depends on Var4
    Var9 = (
        Var7 + Var8 + np.random.normal(0, 1, n_samples)
    )  # Var9 depends on Var7 and Var8

    data = pd.DataFrame(
        {
            "Var0": Var0,
            "Var1": Var1,
            "Var2": Var2,
            "Var3": Var3,
            "Var4": Var4,
            "Var5": Var5,
            "Var6": Var6,
            "Var7": Var7,
            "Var8": Var8,
            "Var9": Var9,
        }
    )

    config.variable_types = VariableTypes(
        continuous=[
            "Var0",
            "Var1",
            "Var2",
            "Var3",
            "Var4",
            "Var5",
            "Var6",
            "Var7",
            "Var8",
            "Var9",
        ]
    )
    config.study_name = "pipe_hard_dataset"

    # Plot all methods with the true causal graph
    compare_pipelines(data, config=config)


def create_true_causal_graph_hard() -> GeneralGraph:
    """
    Creates the true causal graph for the hard dataset using causal-learn's GeneralGraph.

    Returns:
    - GeneralGraph: The true causal graph.
    """
    # Define node names
    node_names = [
        "Var0",
        "Var1",
        "Var2",
        "Var3",
        "Var4",
        "Var5",
        "Var6",
        "Var7",
        "Var8",
        "Var9",
    ]

    # Create GraphNode instances
    nodes = {name: GraphNode(name) for name in node_names}

    # Initialize GeneralGraph
    graph = GeneralGraph(list(nodes.values()))

    # Define true directed edges
    true_edges = [
        ("Var0", "Var2"),
        ("Var1", "Var3"),
        ("Var2", "Var3"),
        ("Var2", "Var4"),
        ("Var3", "Var5"),
        ("Var4", "Var5"),
        ("Var5", "Var6"),
        ("Var1", "Var7"),
        ("Var6", "Var7"),
        ("Var4", "Var8"),
        ("Var7", "Var9"),
        ("Var8", "Var9"),
    ]

    # Add directed edges to the graph
    for source, target in true_edges:
        edge = Edge(
            nodes[source],
            nodes[target],
            Endpoint.TAIL,  # Tail end for the source node
            Endpoint.ARROW,  # Arrow end for the target node
        )
        graph.add_edge(edge)

    return graph
