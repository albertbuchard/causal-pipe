import copy
import itertools
from operator import xor
from typing import List, Tuple

import numpy as np
from bcsl.graph_utils import (
    get_nondirected_edge,
    get_undirected_edge,
    get_directed_edge,
    get_bidirected_edge,
)
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def set_undirected_edges(graph, edges):
    for edge in edges:
        n1 = graph.nodes[edge[0]]
        n2 = graph.nodes[edge[1]]
        edge = graph.get_edge(n1, n2)
        if edge is not None:
            graph.remove_edge(edge)
        edge = graph.get_edge(n2, n1)
        if edge is not None:
            graph.remove_edge(edge)
        graph.add_edge(get_nondirected_edge(n1, n2))
    return graph


def copy_graph(graph: GeneralGraph) -> GeneralGraph:
    nodes = graph.get_nodes()
    copied_graph = GeneralGraph(nodes=nodes)
    edges = graph.get_graph_edges()
    for edge in edges:
        copied_graph.add_edge(edge)
    return copied_graph


def convert_general_graph_to_prior_knowledge(general_graph: GeneralGraph) -> np.ndarray:
    """
    Converts a GeneralGraph instance to a prior knowledge matrix for DirectLiNGAM.

    Parameters
    ----------
    general_graph : GeneralGraph
        The input graph from which to derive prior knowledge.

    Returns
    -------
    prior_knowledge : np.ndarray
        A matrix of shape (n_features, n_features) where:
            - 1 indicates a directed path from i to j
            - 0 indicates no directed path from i to j
            - -1 indicates unknown
    """
    n = general_graph.get_num_nodes()
    prior_knowledge = np.full((n, n), -1, dtype=int)  # Initialize with -1 (unknown)

    node_list = general_graph.get_nodes()

    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            if i == j:
                prior_knowledge[i, j] = 0  # No self-loop
                continue

            if general_graph.is_parent_of(node_i, node_j):
                prior_knowledge[i, j] = 1  # Directed path exists
            elif general_graph.is_child_of(node_i, node_j):
                prior_knowledge[j, i] = 1  # Directed path exists (reverse)
            elif general_graph.is_undirected_from_to(node_i, node_j):
                prior_knowledge[i, j] = -1  # Undirected edge
            else:
                # To determine if no directed path exists, we need to ensure that
                # there is no possible directed path considering the current graph structure.
                # This might require additional logic depending on the graph's properties.
                # For simplicity, we'll set it to 0 if there is no direct evidence of a path.
                prior_knowledge[i, j] = 0

    return prior_knowledge


def general_graph_to_priorknowledge_possibilities(
    general_graph: GeneralGraph,
) -> List[np.ndarray]:
    """
    Converts a GeneralGraph instance to a list of prior knowledge matrices for DirectLiNGAM.
    Each matrix represents a different assignment of directions to undirected edges.

    Parameters
    ----------
    general_graph : GeneralGraph
        The input graph from which to derive prior knowledge matrices.

    Returns
    -------
    List[np.ndarray]
        A list of prior knowledge matrices, each corresponding to a different
        combination of directed edges derived from undirected edges.
    """
    undirected_edges = []
    node_list = general_graph.get_nodes()
    n = general_graph.get_num_nodes()

    # Identify all undirected edges (i < j to avoid duplicates)
    for i in range(n):
        for j in range(i + 1, n):
            node_i = node_list[i]
            node_j = node_list[j]
            if general_graph.is_undirected_from_to(node_i, node_j):
                undirected_edges.append((node_i, node_j))

    k = len(undirected_edges)
    print(f"Found {k} undirected edges. Generating {2**k} prior knowledge matrices.")

    # If there are no undirected edges, return the single prior knowledge matrix
    if k == 0:
        prior_knowledge = convert_general_graph_to_prior_knowledge(general_graph)
        return [prior_knowledge]

    # Generate all possible direction assignments (each undirected edge can be directed in two ways)
    direction_options = list(
        itertools.product([0, 1], repeat=k)
    )  # 0: node_i -> node_j, 1: node_j -> node_i

    prior_knowledges = []

    for idx, option in enumerate(direction_options):
        # Deep copy the original graph to modify it
        new_graph = copy.deepcopy(general_graph)

        # Assign directions based on the current combination
        for edge_idx, direction in enumerate(option):
            node_a, node_b = undirected_edges[edge_idx]
            # Remove the undirected edge first
            new_graph.remove_connecting_edge(node_a, node_b)
            # Add the directed edge based on the direction
            if direction == 0:
                new_graph.add_directed_edge(node_a, node_b)  # node_a -> node_b
            else:
                new_graph.add_directed_edge(node_b, node_a)  # node_b -> node_a

        # Convert the modified graph to a prior knowledge matrix
        prior_knowledge = convert_general_graph_to_prior_knowledge(new_graph)
        prior_knowledges.append(prior_knowledge)

        print(f"Generated prior knowledge matrix {idx + 1}/{len(direction_options)}")

    return prior_knowledges


def edge_b_is_not_an_ancestor_a(edge: Edge) -> bool:
    """
    Checks if node B is not an ancestor of node A.

    Parameters
    ----------
    edge : Edge
        The edge to check.

    Returns
    -------
    bool
        True if edge B is not an ancestor of node A, False otherwise.
    """
    # Enpoint.CIRCLE -> Endpoint.ARROW
    return edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.ARROW


def edge_a_is_not_an_ancestor_b(edge: Edge) -> bool:
    """
    Checks if node A is not an ancestor of node B.

    Parameters
    ----------
    edge : Edge
        The edge to check.

    Returns
    -------
    bool
        True if node A is not an ancestor of node B, False otherwise.

    """
    # Enpoint.CIRCLE -> Endpoint.ARROW
    return edge.endpoint2 == Endpoint.CIRCLE and edge.endpoint1 == Endpoint.ARROW


def edge_with_latent_common_cause(edge: Edge) -> bool:
    """
    Checks if the edge has a latent common cause.

    Parameters
    ----------
    edge : Edge
        The edge to check.

    Returns
    -------
    bool
        True if the edge has a latent common cause, False otherwise.
    """
    # Endpoint.ARROW -> Endpoint.ARROW
    return edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.ARROW


def edge_no_d_separation(edge: Edge) -> bool:
    """
    Checks if there is no d-separation between the nodes connected by the edge.

    Parameters
    ----------
    edge : Edge
        The edge to check.

    Returns
    -------
    bool
        True if there is no d-separation between the nodes, False otherwise.
    """
    # Endpoint.CIRCLE -> Endpoint.CIRCLE
    return edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.CIRCLE


def invert_edge(edge: Edge):
    """
    Inverts the direction of a directed edge.

    Parameters
    ----------
    edge : tuple
        A tuple representing a directed edge (source, target).

    Returns
    -------
    inverted_edge : tuple
        A tuple representing the inverted directed edge (target, source).
    """
    return Edge(
        node1=edge.node2, node2=edge.node1, end1=edge.endpoint1, end2=edge.endpoint2
    )


def general_graph_to_sem_model(general_graph: GeneralGraph) -> str:
    """
    Converts a GeneralGraph instance to a lavaan SEM model string.

    Parameters
    ----------
    general_graph : GeneralGraph
        The input graph from which to derive the SEM model.

    Returns
    -------
    model_str : str
        A string representing the SEM model in lavaan syntax.
    """
    node_list = general_graph.get_nodes()
    node_names = [node.get_name() for node in node_list]
    n = general_graph.get_num_nodes()
    node_map = general_graph.get_node_map()

    # Collect directed edges: list of (source, target)
    directed_edges: List[Tuple[str, str]] = []
    # Collect undirected edges: list of (node1, node2)
    undirected_edges: List[Tuple[str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            node_i = node_list[i]
            node_j = node_list[j]
            edge_i = general_graph.get_edge(node_i, node_j)
            edge_j = general_graph.get_edge(node_j, node_i)
            if not edge_i and not edge_j:
                continue
            # Check if there's a directed edge from node_i to node_j
            if general_graph.is_adjacent_to(node_i, node_j):
                edge = general_graph.get_edge(node_i, node_j)
                edge_node_1 = edge.node1
                edge_node_2 = edge.node2
                node1_name = edge_node_1.get_name()
                node2_name = edge_node_2.get_name()
                if general_graph.is_directed_from_to(edge_node_1, edge_node_2):
                    directed_edges.append((node1_name, node2_name))
                # Check if there's an undirected edge between node_i and node_j
                elif general_graph.is_undirected_from_to(edge_node_1, edge_node_2):
                    # To avoid duplicates, ensure node_i < node_j
                    undirected_edges.append((node1_name, node2_name))
                elif edge_b_is_not_an_ancestor_a(edge):
                    directed_edges.append((node1_name, node2_name))
                elif edge_with_latent_common_cause(edge):
                    undirected_edges.append((node1_name, node2_name))
                elif edge_no_d_separation(edge):
                    undirected_edges.append((node1_name, node2_name))
                else:
                    raise ValueError("Should not reach here.")
            else:
                raise ValueError("Should not reach here.")

    # Group directed edges by target
    regressions = {}
    for source, target in directed_edges:
        if target not in regressions:
            regressions[target] = []
        regressions[target].append(source)

    # Group undirected edges by node1
    residual_covariances = {}
    for node1, node2 in undirected_edges:
        if node1 not in residual_covariances:
            residual_covariances[node1] = []
        residual_covariances[node1].append(node2)

    # Start building the model string
    model_lines = []

    # Add regressions
    if regressions:
        model_lines.append("  # regressions")
        for target, sources in regressions.items():
            sources_str = " + ".join(sources)
            model_lines.append(f"    {target} ~ {sources_str}")
        model_lines.append("")  # Add an empty line for separation

    # Add residual covariances
    if residual_covariances:
        model_lines.append("  # residual covariances")
        for node1, nodes2 in residual_covariances.items():
            nodes2_str = " + ".join(nodes2)
            model_lines.append(f"    {node1} ~~ {nodes2_str}")
        model_lines.append("")  # Add an empty line for separation

    # Combine all lines into a single string
    model_str = "\n".join(model_lines)

    return model_str


def get_all_directed_edges_list(graph: GeneralGraph) -> List[Tuple[int, int]]:
    """
    Get all directed edges in the graph.
    :param graph: GeneralGraph: The graph to get directed edges from.
    """
    directed_edges = []
    node_list = graph.get_nodes()
    n = graph.get_num_nodes()

    for i in range(n):
        for j in range(i + 1, n):
            node_i = node_list[i]
            node_j = node_list[j]
            edge_i = graph.get_edge(node_i, node_j)
            edge_j = graph.get_edge(node_j, node_i)
            if not edge_i and not edge_j:
                continue
            # Check if there's a directed edge from node_i to node_j
            if graph.is_adjacent_to(node_i, node_j):
                edge = graph.get_edge(node_i, node_j)
                edge_node_1 = edge.node1
                edge_node_2 = edge.node2
                if (
                    graph.is_directed_from_to(edge_node_1, edge_node_2)
                    or graph.is_directed_from_to(edge_node_2, edge_node_1)
                    or edge_b_is_not_an_ancestor_a(edge)
                    or edge_a_is_not_an_ancestor_b(edge)
                ):
                    directed_edges.append((i, j))
            else:
                raise ValueError("Should not reach here.")
    return directed_edges


def get_neighbors_general_graph(
    general_graph: GeneralGraph, undirected_edges=None, kept_edges=None
) -> Tuple[List[GeneralGraph], List[Edge]]:
    """
    Generates neighboring graphs by flipping the direction of directed edges
    or converting undirected edges to directed edges in both possible directions.

    Parameters
    ----------
    general_graph : GeneralGraph
        The input graph from which to generate neighbors.
    undirected_edges : List[Tuple[str, str]], optional
        A list of undirected edges to avoid duplicates.
    kept_edges: List[Edge], optional
        A list of edges to keep in the graph.

    Returns
    -------
    Tuple[List[GeneralGraph], List[Edge]]
        A tuple containing the list of neighboring GeneralGraph instances and the list of switched edges.

        neighbors : List[GeneralGraph]
            A list of neighboring GeneralGraph instances.
        switched_edges : List[Edge]
            A list of edges that were switched from directed to undirected or vice versa.
    """
    neighbors = []
    switched_edges = []
    node_list = general_graph.get_nodes()
    n = general_graph.get_num_nodes()
    if undirected_edges is None:
        undirected_edges = []
    if kept_edges is None:
        kept_edges = []

    general_graph = make_edge_undirected_from_edge_list(general_graph, undirected_edges)

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in kept_edges or (j, i) in kept_edges:
                continue
            node_i = node_list[i]
            node_j = node_list[j]
            edge = general_graph.get_edge(node_i, node_j)
            if edge is None:
                continue
            is_directed = is_edge_directed(edge)

            # if (i, j) in undirected_edges or (j, i) in undirected_edges:
            #     continue

            # Check for a directed edge from node_i to node_j
            if is_directed:
                # Create a copy of the graph
                neighbor_graph = switch_directed_edge_in_graph(general_graph, edge)
                neighbors.append(neighbor_graph)
                switched_edges.append(edge)

                # Make undirected edge
                neighbor_graph = make_edge_undirected(general_graph, edge)
                neighbors.append(neighbor_graph)
                switched_edges.append(edge)
            else:
                # Create two neighbors by directing the undirected edge in both possible directions
                # Neighbor 1: node_i -> node_j
                neighbor_graph_1 = copy_graph(general_graph)
                neighbor_graph_1.remove_edge(edge)
                neighbor_graph_1.add_directed_edge(node_i, node_j)
                neighbors.append(neighbor_graph_1)
                switched_edges.append(edge)

                # Neighbor 2: node_j -> node_i
                neighbor_graph_2 = copy_graph(general_graph)
                neighbor_graph_2.remove_edge(edge)
                neighbor_graph_2.add_directed_edge(node_j, node_i)
                neighbors.append(neighbor_graph_2)
                switched_edges.append(edge)

    assert len(neighbors) == len(
        switched_edges
    ), "Number of neighbors and switched edges should be the same."
    return neighbors, switched_edges


def switch_directed_edge_in_graph(graph: GeneralGraph, edge: Edge):
    """
    Switch the direction of the edge in the graph.
    :param graph: GeneralGraph: The graph to modify.
    :param edge: Edge: The edge to switch direction.
    """
    graph = copy_graph(graph)
    node1, node2 = edge.node1, edge.node2
    graph.remove_connecting_edge(node1, node2)
    graph.add_edge(get_directed_edge(node2, node1))
    return graph


def make_edge_undirected(graph: GeneralGraph, edge: Edge):
    """
    Make the edge undirected by removing the directed edge and adding an undirected edge.
    :param graph: GeneralGraph: The graph to modify.
    :param edge: Edge: The directed edge to make undirected.
    """
    graph = copy_graph(graph)
    node1, node2 = edge.node1, edge.node2
    graph.remove_connecting_edge(node1, node2)
    graph.add_edge(get_undirected_edge(node1, node2))
    return graph


def make_edge_undirected_from_edge_list(
    graph: GeneralGraph, edge_list: List[Tuple[int, int]]
):
    """
    Make the edge undirected by removing the directed edge and adding an undirected edge.
    :param graph: GeneralGraph: The graph to modify.
    :param edge_list: List[Tuple[int, int]]: The directed edge to make undirected.
    """
    graph = copy_graph(graph)
    node_list = graph.get_nodes()
    for edge in edge_list:
        node1, node2 = node_list[edge[0]], node_list[edge[1]]
        graph.remove_connecting_edge(node1, node2)
        graph.add_edge(get_undirected_edge(node1, node2))
    return graph


def unify_edge_types_directed_undirected(graph: GeneralGraph) -> GeneralGraph:
    """
    Simplify the edge types in the graph by converting all directed-like edges to directed edges and all undirected-like edges to undirected edges.
    :param graph:  GeneralGraph: The graph to unify.
    :return:  GeneralGraph: The unified graph.
    """
    graph = copy_graph(graph)
    node_list = graph.get_nodes()
    n = graph.get_num_nodes()
    for i in range(n):
        for j in range(i + 1, n):
            node_i = node_list[i]
            node_j = node_list[j]
            edge = graph.get_edge(node_i, node_j)
            if edge is None:
                continue
            is_directed_i_j = graph.is_directed_from_to(
                node_i, node_j
            ) or edge_b_is_not_an_ancestor_a(edge)
            is_directed_j_i = graph.is_directed_from_to(
                node_j, node_i
            ) or edge_a_is_not_an_ancestor_b(edge)
            is_directed = is_directed_i_j or is_directed_j_i
            if is_directed:
                graph.remove_edge(edge)
                graph.add_edge(get_directed_edge(edge.node1, edge.node2))
            else:
                graph.remove_edge(edge)
                graph.add_edge(get_bidirected_edge(edge.node1, edge.node2))
    return graph


import pandas as pd
import numpy as np


def dataframe_to_json_compatible_list(df):
    """
    Converts a pandas DataFrame into a list of rows that are JSON serializable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert.

    Returns:
    --------
    list
        A list of dictionaries representing the DataFrame rows, suitable for JSON serialization.
    """
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Function to convert values to JSON-serializable types
    def convert_value(val):
        if isinstance(val, (np.integer, int)):
            return int(val)
        elif isinstance(val, (np.floating, float)):
            return float(val)
        elif isinstance(val, (np.bool_, bool)):
            return bool(val)
        elif isinstance(val, (np.ndarray, list, tuple)):
            return val.tolist() if isinstance(val, np.ndarray) else list(val)
        elif isinstance(val, (np.datetime64, pd.Timestamp)):
            return val.isoformat() if not pd.isnull(val) else None
        elif pd.isnull(val):
            return None
        else:
            return str(val)  # Convert any other type to string

    # Convert DataFrame to list of dictionaries
    json_compatible_list = []
    for _, row in df.iterrows():
        json_compatible_row = {
            column: convert_value(value) for column, value in row.items()
        }
        json_compatible_list.append(json_compatible_row)

    return json_compatible_list


def is_edge_directed(edge: Edge) -> bool:
    """
    Checks if the edge is directed.

    Parameters
    ----------
    edge : Edge
        The edge to check.

    Returns
    -------
    bool
        True if the edge is directed, False otherwise.
    """
    is_directed_like = edge_a_is_not_an_ancestor_b(edge) or edge_b_is_not_an_ancestor_a(
        edge
    )
    xor_arrow = xor(edge.endpoint1 == Endpoint.ARROW, edge.endpoint2 == Endpoint.ARROW)
    return is_directed_like or xor_arrow


def get_nodes_from_node_names(node_names: List[str]) -> List[GraphNode]:
    """
    Create a list of GraphNode instances from node names.

    Parameters:
    - node_names (List[str]): List of node names.

    Returns:
    - List[GraphNode]: List of GraphNode objects.
    """
    nodes: List[GraphNode] = []
    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)
    return nodes
