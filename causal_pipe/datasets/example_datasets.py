import numpy as np
import pandas as pd
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from causal_pipe.causal_discovery.static_causal_discovery import visualize_graph


def create_true_causal_graph_easy() -> GeneralGraph:
    """
    Creates the true causal graph for the easy dataset using causal-learn's GeneralGraph.

    Returns:
    - GeneralGraph: The true causal graph.
    """
    # Define node names
    node_names = ["Var1", "Var2", "Var3", "Var4", "Var5"]

    # Create GraphNode instances
    nodes = {name: GraphNode(name) for name in node_names}

    # Initialize GeneralGraph
    graph = GeneralGraph(list(nodes.values()))

    # Define true directed edges
    true_edges = [
        ("Var1", "Var3"),
        ("Var2", "Var4"),
        ("Var3", "Var5"),
        ("Var4", "Var5"),
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


def get_easy_df(n_samples):
    true_graph = create_true_causal_graph_easy()
    visualize_graph(true_graph, title="True Causal Graph (EASY)", show=True)

    # Independent variables
    Var1 = np.random.normal(0, 1, n_samples)
    Var2 = np.random.normal(0, 1, n_samples)

    # Dependent variables
    Var3 = 2 * Var1 + np.random.normal(0, 1, n_samples)  # Var3 depends on Var1
    Var4 = 0.5 * Var2 + np.random.normal(0, 1, n_samples)  # Var4 depends on Var2
    Var5 = (
        Var3 + Var4 + np.random.normal(0, 1, n_samples)
    )  # Var5 depends on Var3 and Var4

    # Create DataFrame
    df = pd.DataFrame(
        {"Var1": Var1, "Var2": Var2, "Var3": Var3, "Var4": Var4, "Var5": Var5}
    )

    return df


def create_true_causal_graph_medium() -> GeneralGraph:
    """
    Creates the true causal graph for the medium dataset using causal-learn's GeneralGraph.

    Returns:
    - GeneralGraph: The true causal graph.
    """
    # Define node names for the medium dataset
    node_names = ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7"]

    # Create GraphNode instances
    nodes = {name: GraphNode(name) for name in node_names}

    # Initialize GeneralGraph
    graph = GeneralGraph(list(nodes.values()))

    # Define true directed edges for a more complex structure
    true_edges = [
        ("Var1", "Var3"),  # Var3 depends on Var1
        ("Var2", "Var4"),  # Var4 depends on Var2
        ("Var3", "Var5"),  # Var5 depends on Var3
        ("Var4", "Var5"),  # Var5 also depends on Var4
        ("Var5", "Var6"),  # Var6 depends on Var5
        ("Var6", "Var7"),  # Var7 depends on Var6
        ("Var2", "Var7"),  # Var7 also has an indirect dependency through Var2
        ("Var1", "Var6"),  # Var6 also depends directly on Var1
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


def get_medium_df(n_samples):
    """
    Generate a DataFrame for the medium dataset with the structure as per the
    medium causal graph.

    Parameters:
    - n_samples: Number of samples to generate

    Returns:
    - pd.DataFrame: Generated data with the defined causal structure
    """

    true_graph = create_true_causal_graph_medium()
    visualize_graph(true_graph, title="True Causal Graph (MEDIUM)", show=True)

    # Independent variables
    Var1 = np.random.normal(0, 1, n_samples)  # Independent Var1
    Var2 = np.random.normal(0, 1, n_samples)  # Independent Var2

    # Dependent variables
    Var3 = 2 * Var1 + np.random.normal(0, 1, n_samples)  # Var3 depends on Var1
    Var4 = 0.5 * Var2 + np.random.normal(0, 1, n_samples)  # Var4 depends on Var2
    Var5 = (
        1.5 * Var3 + 0.8 * Var4 + np.random.normal(0, 1, n_samples)
    )  # Var5 depends on Var3 and Var4
    Var6 = (
        1.2 * Var5 + 0.6 * Var1 + np.random.normal(0, 1, n_samples)
    )  # Var6 depends on Var5 and Var1
    Var7 = (
        1.1 * Var6 + 0.7 * Var2 + np.random.normal(0, 1, n_samples)
    )  # Var7 depends on Var6 and Var2

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Var1": Var1,
            "Var2": Var2,
            "Var3": Var3,
            "Var4": Var4,
            "Var5": Var5,
            "Var6": Var6,
            "Var7": Var7,
        }
    )

    return df


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


def get_hard_dataset(n_samples):
    # Generate synthetic data for testing
    np.random.seed(42)

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

    return data
