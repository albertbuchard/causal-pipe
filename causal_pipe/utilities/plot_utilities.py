import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_correlation_graph(
    corr_matrix,
    labels=None,
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
    min_edge_width=1,
    max_edge_width=5,
    title=None,
    node_order=None,
    auto_order=False,
    edge_width=None,
    node_kwargs=None,
    edge_kwargs=None,
    output_path=None,
    show=True,
):
    """
    Plot a correlation graph using NetworkX.

    Parameters:
    - corr_matrix (np.ndarray): The correlation or partial correlation matrix.
    - labels (list): Variable names corresponding to the matrix.
    - threshold (float): Minimum absolute value for edges to be included.
    - layout (str): Layout for the graph ('spring', 'circular', 'hierarchical', 'grid').
    - node_size (int): Size of the nodes.
    - node_color (str or list): Color of the nodes.
    - font_size (int): Font size for labels.
    - edge_cmap (str): Colormap for the edges.
    - edge_vmin (float): Minimum value for edge colormap.
    - edge_vmax (float): Maximum value for edge colormap.
    - with_edge_labels (bool): Whether to display edge labels.
    - figsize (tuple): Figure size.
    - min_edge_width (float): Minimum edge width.
    - max_edge_width (float): Maximum edge width.
    - title (str): Title of the plot.
    - node_order (list): Predefined ordering of nodes (used in 'hierarchical' and 'grid' layouts).
    - auto_order (bool): Automatically order nodes based on number of edges over threshold.
    - edge_width (float or list): Custom edge width or list of widths.
    - node_kwargs (dict): Additional keyword arguments for drawing nodes.
    - edge_kwargs (dict): Additional keyword arguments for drawing edges.
    """
    if node_kwargs is None:
        node_kwargs = {}
    if edge_kwargs is None:
        edge_kwargs = {}

    if isinstance(corr_matrix, pd.DataFrame):
        if labels is None:
            labels = corr_matrix.columns
        corr_matrix = corr_matrix.values

    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    G = nx.Graph()

    if labels is None:
        labels = [f"Var{i}" for i in range(corr_matrix.shape[0])]

    # Add nodes
    G.add_nodes_from(labels)

    # Add edges with weights above the threshold
    if threshold is None:
        threshold = -np.inf
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            weight = corr_matrix[i, j]
            if abs(weight) >= threshold:
                G.add_edge(labels[i], labels[j], weight=weight)

    # Automatically define node_order based on degree if auto_order is True
    if auto_order:
        # Calculate degrees (number of edges over threshold)
        degrees = dict(G.degree())
        # Sort nodes by degree (descending order)
        node_order = sorted(degrees, key=lambda x: degrees[x], reverse=True)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        # Use the undirected graph G for hierarchical layout
        if node_order is None or len(node_order) == 0:
            # Use node with highest degree as root
            degrees = dict(G.degree())
            root_node = max(degrees, key=degrees.get)
        else:
            root_node = node_order[0]

        # Use graphviz_layout with 'dot' for hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", root=root_node)
        except (ImportError, nx.NetworkXException):
            # Fallback to spring layout if graphviz is not available
            print("Graphviz layout not available, using spring layout")
            pos = nx.spring_layout(G, seed=42)
    elif layout == "grid":
        if node_order is None:
            node_order = labels
        # Arrange nodes in a grid
        sqrt_n = int(np.ceil(np.sqrt(len(node_order))))
        grid_positions = {}
        for idx, node in enumerate(node_order):
            row = idx // sqrt_n
            col = idx % sqrt_n
            grid_positions[node] = (col, -row)
        pos = grid_positions
    else:
        raise ValueError(
            "Invalid layout. Choose 'spring', 'circular', 'hierarchical', or 'grid'."
        )

    # Get edge weights for coloring and widths
    edges = G.edges(data=True)
    edge_weights = []
    for u, v, data in edges:
        if "weight" in data:
            edge_weights.append(data["weight"])
        else:
            edge_weights.append(0)  # Default weight if missing

    if edge_width is None:
        # Normalize edge widths based on correlation magnitude
        abs_weights = [abs(w) for w in edge_weights]
        min_weight = min(abs_weights) if abs_weights else 0
        max_weight = max(abs_weights) if abs_weights else 1  # Avoid division by zero
        if max_weight == min_weight:
            widths = [max_edge_width] * len(abs_weights)
        else:
            widths = [
                min_edge_width
                + (abs(w) - min_weight)
                / (max_weight - min_weight)
                * (max_edge_width - min_edge_width)
                for w in abs_weights
            ]
    else:
        widths = edge_width

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_color, **node_kwargs
    )
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold")

    # Draw edges (undirected)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_weights,
        edge_cmap=plt.get_cmap(edge_cmap),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=widths,
        **edge_kwargs,
    )

    # Draw edge labels if required
    if with_edge_labels:
        edge_labels = {}
        for u, v, data in edges:
            weight = data.get("weight", 0)
            edge_labels[(u, v)] = f"{weight:.2f}"
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=font_size - 2
        )

    plt.axis("off")
    if title:
        plt.title(title, fontsize=font_size + 2)
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()


def plot_correlation_graph_bak(
    corr_matrix,
    labels=None,
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
    min_edge_width=1,
    max_edge_width=5,
    title=None,
    node_order=None,
    auto_order=False,
    edge_width=None,
    **kwargs,
):
    """
    Plot a correlation graph using NetworkX.

    Parameters:
    - corr_matrix (np.ndarray): The correlation or partial correlation matrix.
    - labels (list): Variable names corresponding to the matrix.
    - threshold (float): Minimum absolute value for edges to be included.
    - layout (str): Layout for the graph ('spring', 'circular', 'tree', 'grid').
    - node_size (int): Size of the nodes.
    - node_color (str or list): Color of the nodes.
    - font_size (int): Font size for labels.
    - edge_cmap (str): Colormap for the edges.
    - edge_vmin (float): Minimum value for edge colormap.
    - edge_vmax (float): Maximum value for edge colormap.
    - with_edge_labels (bool): Whether to display edge labels.
    - figsize (tuple): Figure size.
    - min_edge_width (float): Minimum edge width.
    - max_edge_width (float): Maximum edge width.
    - title (str): Title of the plot.
    - node_order (list): Predefined ordering of nodes (used in 'tree' and 'grid' layouts).
    - auto_order (bool): Automatically order nodes based on number of edges over threshold.
    - **kwargs: Additional keyword arguments for NetworkX drawing functions.
    """
    G = nx.Graph()

    if labels is None:
        labels = [f"Var{i}" for i in range(corr_matrix.shape[0])]

    # Add nodes
    G.add_nodes_from(labels)

    # Add edges with weights above the threshold
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            weight = corr_matrix[i, j]
            if abs(weight) >= threshold:
                G.add_edge(labels[i], labels[j], weight=weight)

    # Automatically define node_order based on degree if auto_order is True
    if auto_order:
        # Calculate degrees (number of edges over threshold)
        degrees = dict(G.degree())
        # Sort nodes by degree (descending order)
        node_order = sorted(degrees, key=lambda x: degrees[x], reverse=True)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "tree":
        if node_order is None:
            node_order = labels
        # Assign layers based on node_order
        layers = {node: idx for idx, node in enumerate(node_order)}
        nx.set_node_attributes(G, layers, "layer")
        pos = nx.multipartite_layout(G, subset_key="layer")
    elif layout == "grid":
        if node_order is None:
            node_order = labels
        sqrt_n = int(np.ceil(np.sqrt(len(node_order))))
        grid_positions = {}
        for idx, node in enumerate(node_order):
            row = idx // sqrt_n
            col = idx % sqrt_n
            grid_positions[node] = (col, -row)
        pos = grid_positions
    else:
        raise ValueError(
            "Invalid layout. Choose 'spring', 'circular', 'tree', or 'grid'."
        )

    # Get edge weights for coloring and widths
    edges = G.edges(data=True)
    edge_weights = [data["weight"] for _, _, data in edges]
    if edge_width is None:
        # Normalize edge widths based on correlation magnitude
        abs_weights = [abs(w) for w in edge_weights]
        min_weight = min(abs_weights) if abs_weights else 0
        max_weight = max(abs_weights) if abs_weights else 1  # Avoid division by zero
        # Avoid division by zero
        if max_weight == min_weight:
            widths = [max_edge_width] * len(abs_weights)
        else:
            widths = [
                min_edge_width
                + (abs(w) - min_weight)
                / (max_weight - min_weight)
                * (max_edge_width - min_edge_width)
                for w in abs_weights
            ]
    else:
        widths = edge_width

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, **kwargs)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_weights,
        edge_cmap=plt.get_cmap(edge_cmap),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=widths,
    )
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold")

    # Draw edge labels if required
    if with_edge_labels:
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=font_size - 2
        )

    plt.axis("off")
    if title:
        plt.title(title, fontsize=font_size + 2)
    plt.show()
