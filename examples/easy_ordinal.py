# examples/easy_with_ordinal.py

import numpy as np
import pandas as pd
from typing import List

from bcsl.graph_utils import visualize_graph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from causal_pipe.causal_pipe import CausalPipeConfig
from causal_pipe.pipe_config import VariableTypes, CausalEffectMethod, FASSkeletonMethod
from examples.utilities import compare_pipelines


def compare_easy_dataset_with_ordinal(config: CausalPipeConfig) -> None:
    """
    Generates a synthetic dataset with at least two ordinal variables,
    defines the true causal graph, visualizes it, and compares different
    causal discovery pipelines.

    Parameters:
    - config (CausalPipeConfig): Configuration object for the causal pipeline.
    """
    # Set random seed for reproducibility
    np.random.seed(config.seed)

    # Define and visualize the true causal graph
    true_graph = create_true_causal_graph_easy_with_ordinal()
    visualize_graph(
        true_graph, title="True Causal Graph (EASY with Ordinal)", show=True
    )

    n_samples = 500  # Number of samples in the synthetic dataset

    # Generate independent continuous variable Var0
    Var0 = np.random.normal(0, 1, n_samples)

    # Generate Var1 and Var2 as continuous variables influenced by Var0
    Var1_continuous = np.random.normal(0, 1, n_samples) + Var0 * 3
    Var2_continuous = np.random.normal(0, 1, n_samples) + Var0
    Ord3_continuous = np.random.normal(0, 1, n_samples)

    # Discretize Var1 and Var2 into ordinal categories: Low, Medium, High
    Ord1 = pd.cut(
        Var1_continuous,
        bins=3,
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).codes  # Convert categories to integer codes

    Ord2 = pd.cut(
        Var2_continuous,
        bins=3,
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).codes  # Convert categories to integer codes

    Ord3 = pd.cut(
        Ord3_continuous,
        bins=3,
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).codes  # Convert categories to integer codes

    # Generate dependent continuous variables Var3 and Var4
    Var3 = -2 * Ord1 + np.random.normal(0, 1, n_samples)  # Var3 depends on Ord1
    Var4 = (
        0.5 * Ord2 + Ord3 + np.random.normal(0, 1, n_samples)
    )  # Var4 depends on Ord2 and Ord3

    # Generate Var5 as a dependent variable influenced by Var3 and Var4
    Var5 = (
        Var3 + Var4 + np.random.normal(0, 1, n_samples)
    )  # Var5 depends on Var3 and Var4

    # Unrelated variable
    Var6 = np.random.normal(0, 1, n_samples)

    # Create a DataFrame with the generated variables
    data = pd.DataFrame(
        {
            "Var0": Var0,
            "Ord1": Ord1,  # Ordinal variable
            "Ord2": Ord2,  # Ordinal variable
            "Ord3": Ord3,  # Ordinal variable
            "Var3": Var3,
            "Var4": Var4,
            "Var5": Var5,
            "Var6": Var6,
        }
    )

    # Update the configuration to specify variable types
    config.variable_types = VariableTypes(
        continuous=["Var0", "Var3", "Var4", "Var5", "Var6"],
        ordinal=["Ord1", "Ord2", "Ord3"],
    )

    config.study_name = "pipe_easy_dataset_with_ordinal"
    config.causal_effect_methods = [
        # For ordinal data
        CausalEffectMethod(name="sem", directed=True)
    ]
    config.preprocessing_params.keep_only_correlated_with = ["Var0", "Var5"]
    config.preprocessing_params.filter_method = "spearman"

    # Compare different causal discovery pipelines using the generated data and configuration
    compare_pipelines(data, config=config)


def create_true_causal_graph_easy_with_ordinal() -> GeneralGraph:
    """
    Creates the true causal graph for the easy dataset with ordinal variables
    using causal-learn's GeneralGraph.

    Returns:
    - GeneralGraph: The true causal graph.
    """
    # Define node names
    node_names = ["Var0", "Ord1", "Ord2", "Ord3", "Var3", "Var4", "Var5", "Var6"]

    # Create GraphNode instances for each variable
    nodes = {name: GraphNode(name) for name in node_names}

    # Initialize the GeneralGraph with the created nodes
    graph = GeneralGraph(list(nodes.values()))

    # Define true directed edges based on the causal relationships
    true_edges = [
        ("Var0", "Ord1"),
        ("Var0", "Ord2"),
        ("Ord1", "Var3"),
        ("Ord2", "Var4"),
        ("Ord3", "Var4"),
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
