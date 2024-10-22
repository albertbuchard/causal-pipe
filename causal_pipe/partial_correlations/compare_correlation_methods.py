from bcsl.bcsl import BCSL
from causallearn.search.ConstraintBased.FCI import fci

from causal_pipe.partial_correlations.partial_correlations import (
    compute_correlations,
    compute_partial_correlations,
    compute_partial_correlations_corrected,
    fci_orient_edges_from_adjacency_and_sepsets,
)
from causal_pipe.causal_discovery.static_causal_discovery import (
    visualize_graph,
    run_causal_discovery,
    visualize_causal_graph,
)
from causal_pipe.utilities.plot_utilities import plot_correlation_graph


def _compare_correlation_methods(df):
    """
    Compare different correlation methods using a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input data.
    """
    # Compute Pearson correlation matrix
    corr_matrix = compute_correlations(df)

    # Compute partial correlations
    partial_corr_matrix = compute_partial_correlations(df)

    # Plot settings
    plot_settings = {
        "labels": df.columns,
        "threshold": 0.2,
        "layout": "spring",
        "node_size": 2500,
        # "node_color": "lightgreen",
        "font_size": 12,
        "edge_cmap": "bwr",
        "edge_vmin": -1,
        "edge_vmax": 1,
        "with_edge_labels": True,
        "figsize": (8, 8),
        "edge_width": 2.5,
    }

    graph_fci, edges_fci = fci(
        dataset=df.values,
        independence_test_method="fisherz",
    )
    visualize_graph(
        graph_fci,
        title="Causal Learn FCI Fischerz Result with FAS for Markov Blanket",
        labels=df.columns,
    )

    # BCSL
    bcsl = BCSL(
        data=df,
        num_bootstrap_samples=100,
        conditional_independence_method="fisherz",
        orientation_method="fci",
        multiple_comparison_correction="fdr",
    )
    undirected_graph = bcsl.combine_local_to_global_skeleton(bootstrap_all_edges=True)
    print("Global Skeleton (resolved):", bcsl.global_skeleton)
    visualize_graph(undirected_graph, title="BCSL Global Skeleton", show=True)
    dag = bcsl.orient_edges(method="fci")
    print("Final DAG:", dag)
    visualize_graph(dag, title="Final DAG (BCSL FCI)", show=True)
    dag = bcsl.orient_edges(method="hill_climbing")
    print("Final DAG hill_climbing:", dag)
    visualize_graph(dag, title="Final DAG (BCSL Hill Climbing)", show=True)

    graph_fci, edges_fci = fci(dataset=df.values, independence_test_method="kci")
    visualize_graph(
        graph_fci,
        title="Causal Learn FCI KCI Result with FAS for Markov Blanket",
        labels=df.columns,
    )

    # Run causal discovery algorithm
    method = "fci"
    result = run_causal_discovery(df, method=method, verbose=True)

    # Visualize the causal graph
    labels = df.columns.tolist()
    visualize_causal_graph(
        result, title=f"{method.upper()} Result, no cross-validation", labels=labels
    )

    partial_corr_matrix_corrected, sepsets = compute_partial_correlations_corrected(
        df, method="glasso", refine_sepsets=False
    )

    plot_correlation_graph(
        partial_corr_matrix_corrected,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Partial Correlation Graph Corrected for Collider Bias",
    )

    partial_corr_matrix_corrected, sepsets = compute_partial_correlations_corrected(
        df, method="glasso", refine_sepsets=True
    )

    # plot_correlation_graph(
    #     partial_corr_matrix_corrected,
    #     labels=df.columns,
    #     threshold=0.01,
    #     layout="hierarchical",
    #     auto_order=True,
    #     node_size=2500,
    #     node_color="lightblue",
    #     font_size=12,
    #     edge_cmap="bwr",
    #     edge_vmin=-1,
    #     edge_vmax=1,
    #     min_edge_width=1,
    #     max_edge_width=5,
    #     title="Partial Correlation Graph Corrected for Collider Bias - Refined Sepsets",
    # )

    graph, edges = fci_orient_edges_from_adjacency_and_sepsets(
        data=df,
        adjacency_matrix=partial_corr_matrix_corrected,
        sepsets=sepsets,
        node_names=df.columns,
        independence_test_method="fisherz",
        verbose=True,
    )

    visualize_graph(
        graph, title="Corrected Glasso for MB + FCI orientation", labels=df.columns
    )
    return result

    # Plot the Pearson correlation graph
    plot_correlation_graph(
        corr_matrix, title="Pearson Correlation Graph", **plot_settings
    )

    plot_correlation_graph(
        corr_matrix,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Pearson Correlation Graph with Automatic Node Ordering",
    )

    # Plot the Partial Correlation graph
    plot_correlation_graph(
        partial_corr_matrix,
        title="Partial Correlation Graph",
        auto_order=True,
        **plot_settings,
    )

    plot_correlation_graph(
        partial_corr_matrix,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Partial Correlation Graph with Automatic Node Ordering",
    )

    partial_corr_matrix_pearson = compute_partial_correlations(
        df,
        method="pearson",
    )

    plot_correlation_graph(
        partial_corr_matrix_pearson,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Pearson Partial Correlation Graph",
    )

    partial_corr_matrix_spearman = compute_partial_correlations(
        df,
        method="spearman",
    )

    plot_correlation_graph(
        partial_corr_matrix_spearman,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Spearman Partial Correlation Graph",
    )

    partial_corr_matrix_nonlinear = compute_partial_correlations(df, method="kci")
    plot_correlation_graph(
        partial_corr_matrix_nonlinear,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Non-linear Partial Correlation Graph",
    )

    partial_corr_matrix_nonlinear_corrected, _ = compute_partial_correlations_corrected(
        df,
        method="kci",
        correction_method="kci",
        verbose=True,
    )

    plot_correlation_graph(
        partial_corr_matrix_nonlinear_corrected,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="Non-linear Partial Correlation Graph Corrected for Collider Bias Non-linearly",
    )

    partial_corr_matrix_nonparametric = compute_partial_correlations(
        df, method="conditional_mi"
    )

    plot_correlation_graph(
        partial_corr_matrix_nonparametric,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="MI Non-parametric Partial Correlation Graph",
    )

    partial_corr_matrix_nonparametric_corrected = (
        compute_partial_correlations_corrected(
            df,
            method="conditional_mi",
            correction_method="conditional_mi",
            verbose=True,
        )
    )

    plot_correlation_graph(
        partial_corr_matrix_nonparametric_corrected,
        labels=df.columns,
        threshold=0.01,
        layout="hierarchical",
        auto_order=True,
        node_size=2500,
        node_color="lightblue",
        font_size=12,
        edge_cmap="bwr",
        edge_vmin=-1,
        edge_vmax=1,
        min_edge_width=1,
        max_edge_width=5,
        title="MI Non-parametric Partial Correlation Graph Corrected for Collider Bias Non-linearly",
    )
