import networkx as nx

import numpy as np
import pandas as pd

from causal_pipe.partial_correlations.compare_correlation_methods import (
    _compare_correlation_methods,
)
from causal_pipe.datasets.example_datasets import (
    get_medium_df,
    get_hard_dataset,
    get_easy_df,
)
from causal_pipe.partial_correlations.partial_correlations import (
    compute_partial_correlations,
)
from causal_pipe.utilities.plot_utilities import plot_correlation_graph

if __name__ == "__main__":
    # Test with pygraphviz
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        method = "pygraphviz"
    except ImportError:
        # Test with pydot
        try:
            from networkx.drawing.nx_pydot import graphviz_layout

            method = "pydot"
        except ImportError:
            graphviz_layout = None

    if graphviz_layout is None:
        print("graphviz_layout is not available.")
    else:
        print(f"graphviz_layout is available using {method}.")

        # Create a simple graph and test layout
        G = nx.complete_graph(5)
        pos = graphviz_layout(G, prog="dot")
        print("graphviz_layout is working.")

    # Generate some random data
    np.random.seed(42)
    n_samples, n_features = 100, 5
    data = np.random.randn(n_samples, n_features)
    df = pd.DataFrame(data, columns=[f"Var{i}" for i in range(n_features)])

    # Compute partial correlation matrix
    partial_corr_matrix = compute_partial_correlations(df, method="glasso")

    # Plot the partial correlation graph
    plot_correlation_graph(
        partial_corr_matrix,
        labels=df.columns,
        threshold=0.1,
        layout="spring",
        node_size=3000,
        node_color="skyblue",
        font_size=12,
        edge_cmap="coolwarm",
        edge_vmin=-1,
        edge_vmax=1,
        with_edge_labels=True,
        figsize=(10, 10),
        edge_width=2,
    )

    n_samples = 200

    easy_df = get_easy_df(n_samples)
    _compare_correlation_methods(easy_df)

    medium_df = get_medium_df(n_samples)
    _compare_correlation_methods(medium_df)

    hard_df = get_hard_dataset(n_samples)
    _compare_correlation_methods(hard_df)
