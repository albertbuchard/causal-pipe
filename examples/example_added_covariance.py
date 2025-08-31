import numpy as np
import pandas as pd

from causal_pipe.causal_discovery.static_causal_discovery import visualize_graph
from causallearn.graph.GeneralGraph import GeneralGraph
from causal_pipe.causal_pipe import (
    CausalPipe,
    CausalPipeConfig,
    FASSkeletonMethod,
    FCIOrientationMethod,
)
from causal_pipe.pipe_config import (
    VariableTypes,
    DataPreprocessingParams,
    CausalEffectMethod,
)


def simulate_data(n_samples: int = 500, seed: int = 0) -> pd.DataFrame:
    """Simulate dataset with a residual covariance between Var0 and Ord3."""
    rng = np.random.default_rng(seed)

    # Define and visualize the true causal graph
    true_graph = create_true_causal_graph_easy_with_ordinal_and_resid_cov()
    visualize_graph(
        true_graph, title="True Causal Graph (EASY with Ordinal) and Residual Covariance", show=True
    )

    # Var0 and Ord3 share unmeasured influence (correlated residuals)
    cov = 0.5
    var0, ord3_cont = rng.multivariate_normal(
        [0.0, 0.0], [[1.0, cov], [cov, 1.0]], size=n_samples
    ).T

    # Other variables following the easy_ordinal example
    var1_cont = rng.normal(size=n_samples) + var0 * 3
    var2_cont = rng.normal(size=n_samples) + var0

    ord1 = pd.cut(var1_cont, bins=3, labels=["Low", "Medium", "High"], include_lowest=True).codes
    ord2 = pd.cut(var2_cont, bins=3, labels=["Low", "Medium", "High"], include_lowest=True).codes
    ord3 = pd.cut(ord3_cont, bins=3, labels=["Low", "Medium", "High"], include_lowest=True).codes

    var3 = -2 * ord1 + rng.normal(size=n_samples)
    var4 = 0.5 * ord2 + ord3 + rng.normal(size=n_samples)
    var5 = var3 + var4 + rng.normal(size=n_samples)
    var6 = rng.normal(size=n_samples)

    return pd.DataFrame(
        {
            "Var0": var0,
            "Ord1": ord1,
            "Ord2": ord2,
            "Ord3": ord3,
            "Var3": var3,
            "Var4": var4,
            "Var5": var5,
            "Var6": var6,
        }
    )


def create_true_causal_graph_easy_with_ordinal_and_resid_cov() -> GeneralGraph:
    """
    Creates the true causal graph for the easy dataset with ordinal variables
    and a residual covariance between Var0 and Ord3 using causal-learn's GeneralGraph.

    Returns:
    - GeneralGraph: The true causal graph.
    """
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode

    # Define node names
    nodes = {
        "Var0": GraphNode("Var0"),
        "Ord1": GraphNode("Ord1"),
        "Ord2": GraphNode("Ord2"),
        "Ord3": GraphNode("Ord3"),
        "Var3": GraphNode("Var3"),
        "Var4": GraphNode("Var4"),
        "Var5": GraphNode("Var5"),
        "Var6": GraphNode("Var6"),
    }

    # Create the graph
    graph = GeneralGraph(list(nodes.values()))

    # Define true directed edges
    directed_edges = [
        ("Var0", "Ord1"),
        ("Var0", "Ord2"),
        ("Ord1", "Var3"),
        ("Ord2", "Var4"),
        ("Ord3", "Var4"),
        ("Var3", "Var5"),
        ("Var4", "Var5"),
    ]
    for source, target in directed_edges:
        edge = Edge(
            nodes[source],
            nodes[target],
            Endpoint.TAIL,  # Tail end for the source node
            Endpoint.ARROW,  # Arrow end for the target node
        )
        graph.add_edge(edge)

    # Define the residual covariance (bidirected edge) between Var0 and Ord3
    edges = [
        Edge(
            nodes["Var0"],
            nodes["Ord3"],
            Endpoint.ARROW,  # Arrow end for Var0
            Endpoint.ARROW,  # Arrow end for Ord3
        )
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


def run_pipeline_with_resid_covariance():
    data = simulate_data()

    config = CausalPipeConfig(
        variable_types=VariableTypes(
            continuous=["Var0", "Var3", "Var4", "Var5", "Var6"],
            ordinal=["Ord1", "Ord2", "Ord3"],
        ),
        preprocessing_params=DataPreprocessingParams(),
        skeleton_method=FASSkeletonMethod(),
        orientation_method=FCIOrientationMethod(),
        causal_effect_methods=[
            CausalEffectMethod(
                name="sem-climbing",
                directed=True,
                params={
                    "estimator": "MLR",
                    "finalize_with_resid_covariances": True,
                    "whitelist_pairs": pd.DataFrame({"lhs": ["Var0"], "rhs": ["Ord3"]}),
                    "max_add": 1,
                    "max_iter": 2,
                },
            )
        ],
        study_name="example_added_covariance",
        output_path="./output/",
        show_plots=False,
        verbose=True,
    )

    pipe = CausalPipe(config)
    pipe.run_pipeline(data)

    aug = pipe.causal_effects["sem-climbing"].get("resid_cov_aug")
    print("Added covariances:", aug["added_covariances"])
    print("Initial fit:", aug["initial_fit_measures"])
    print("Final fit:", aug["fit_measures"])
    print("Final model string:\n", aug["final_model_string"])


if __name__ == "__main__":
    run_pipeline_with_resid_covariance()
