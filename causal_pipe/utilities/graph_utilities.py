import copy
import itertools
from operator import xor
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pydot
import warnings
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
from causallearn.graph.NodeType import NodeType


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


def is_fully_oriented(edge: Edge) -> bool:
    """Return True if the edge has no circle endpoints (e.g., →, ↔, or —)."""
    return edge.endpoint1 != Endpoint.CIRCLE and edge.endpoint2 != Endpoint.CIRCLE


def both_circles(edge: Edge) -> bool:
    """Return True if both endpoints of the edge are circles."""
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


def general_graph_to_sem_model(general_graph: GeneralGraph) -> Tuple[str, List[str]]:
    """
    Converts a GeneralGraph instance to a lavaan SEM model string.
    Adds variance to exogenous variables and residual covariances for undirected edges.

    Parameters
    ----------
    general_graph : GeneralGraph
        The input graph from which to derive the SEM model.

    Returns
    -------
    model_str : str
        A string representing the SEM model in lavaan syntax.
    exogenous_variables : List[str]
        A list of exogenous variables (nodes with no incoming edges).
    """
    node_list = general_graph.get_nodes()
    node_names = [node.get_name() for node in node_list]
    exogenous_variables = set(node_names)
    n = general_graph.get_num_nodes()
    node_map = general_graph.get_node_map()

    # Collect directed edges: list of (source, target)
    directed_edges: List[Tuple[str, str]] = []
    # Collect undirected edges: list of (node1, node2)
    undirected_edges: List[Tuple[str, str]] = []

    seen_nodes = set()
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
                seen_nodes.add(node1_name)
                seen_nodes.add(node2_name)
                if general_graph.is_directed_from_to(edge_node_1, edge_node_2):
                    directed_edges.append((node1_name, node2_name))
                    exogenous_variables.discard(node2_name)
                # Check if there's an undirected edge between node_i and node_j
                elif general_graph.is_undirected_from_to(edge_node_1, edge_node_2):
                    # To avoid duplicates, ensure node_i < node_j
                    undirected_edges.append((node1_name, node2_name))
                    exogenous_variables.discard(node2_name)
                    exogenous_variables.discard(node1_name)
                elif edge_b_is_not_an_ancestor_a(edge):
                    directed_edges.append((node1_name, node2_name))
                    exogenous_variables.discard(node2_name)
                elif edge_with_latent_common_cause(edge):
                    undirected_edges.append((node1_name, node2_name))
                    exogenous_variables.discard(node2_name)
                    exogenous_variables.discard(node1_name)
                elif edge_no_d_separation(edge):
                    undirected_edges.append((node1_name, node2_name))
                    exogenous_variables.discard(node2_name)
                    exogenous_variables.discard(node1_name)
                else:
                    raise ValueError("Should not reach here.")
            else:
                raise ValueError("Should not reach here.")

    # Only keep exogenous variables that are in the graph
    exogenous_variables = list(exogenous_variables.intersection(seen_nodes))

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

    # Add variances for exogenous variables
    if exogenous_variables:
        model_lines.append("  # variances")
        for exog in exogenous_variables:
            model_lines.append(f"    {exog} ~~ {exog}")
        model_lines.append("")  # Add an empty line for separation

    # Combine all lines into a single string
    model_str = "\n".join(model_lines)

    return model_str, exogenous_variables


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
    general_graph: GeneralGraph,
    undirected_edges=None,
    kept_edges=None,
    *,
    respect_pag: bool = False,
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

    if not respect_pag:
        general_graph = make_edge_undirected_from_edge_list(
            general_graph, undirected_edges
        )

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in kept_edges or (j, i) in kept_edges:
                continue
            node_i = node_list[i]
            node_j = node_list[j]
            edge = general_graph.get_edge(node_i, node_j)
            if edge is None:
                continue

            if respect_pag:
                if is_fully_oriented(edge):
                    continue

                if both_circles(edge):
                    neighbor_graph_1 = copy_graph(general_graph)
                    neighbor_graph_1.remove_edge(edge)
                    neighbor_graph_1.add_edge(get_directed_edge(node_i, node_j))
                    neighbors.append(neighbor_graph_1)
                    switched_edges.append(edge)

                    neighbor_graph_2 = copy_graph(general_graph)
                    neighbor_graph_2.remove_edge(edge)
                    neighbor_graph_2.add_edge(get_directed_edge(node_j, node_i))
                    neighbors.append(neighbor_graph_2)
                    switched_edges.append(edge)
                    continue

                if edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.ARROW:
                    neighbor_graph = copy_graph(general_graph)
                    neighbor_graph.remove_edge(edge)
                    neighbor_graph.add_edge(get_directed_edge(node_i, node_j))
                    neighbors.append(neighbor_graph)
                    switched_edges.append(edge)
                    continue

                if edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.CIRCLE:
                    neighbor_graph = copy_graph(general_graph)
                    neighbor_graph.remove_edge(edge)
                    neighbor_graph.add_edge(get_directed_edge(node_j, node_i))
                    neighbors.append(neighbor_graph)
                    switched_edges.append(edge)
                    continue

                if edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.TAIL:
                    neighbor_graph = copy_graph(general_graph)
                    neighbor_graph.remove_edge(edge)
                    neighbor_graph.add_edge(get_directed_edge(node_j, node_i))
                    neighbors.append(neighbor_graph)
                    switched_edges.append(edge)
                    continue

                if edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.CIRCLE:
                    neighbor_graph = copy_graph(general_graph)
                    neighbor_graph.remove_edge(edge)
                    neighbor_graph.add_edge(get_directed_edge(node_i, node_j))
                    neighbors.append(neighbor_graph)
                    switched_edges.append(edge)
                    continue

                continue

            is_directed = is_edge_directed(edge)

            # Check for a directed edge from node_i to node_j
            if is_directed:
                neighbor_graph = switch_directed_edge_in_graph(general_graph, edge)
                neighbors.append(neighbor_graph)
                switched_edges.append(edge)

                neighbor_graph = make_edge_undirected(general_graph, edge)
                neighbors.append(neighbor_graph)
                switched_edges.append(edge)
            else:
                neighbor_graph_1 = copy_graph(general_graph)
                neighbor_graph_1.remove_edge(edge)
                neighbor_graph_1.add_directed_edge(node_i, node_j)
                neighbors.append(neighbor_graph_1)
                switched_edges.append(edge)

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
                if edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
                    graph.add_edge(get_directed_edge(edge.node1, edge.node2))
                elif edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL:
                    graph.add_edge(get_directed_edge(edge.node2, edge.node1))
                elif is_directed_i_j:
                    graph.add_edge(get_directed_edge(node_i, node_j))
                else:
                    graph.add_edge(get_directed_edge(node_j, node_i))
            else:
                graph.remove_edge(edge)
                graph.add_edge(get_bidirected_edge(node_i, node_j))
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


def add_edge_coefficients_from_sem_fit(
    graph: GeneralGraph, model_output: Dict[str, Any]
):
    """
    Validates that all edges in the graph exist in the model and adds coefficient information.

    Parameters:
    - graph: An instance of GeneralGraph.
    - model_output: A dictionary containing the model output, including 'structural_model',
      'residual_covariances', and 'measurement_model'.
    """
    # Convert graph
    graph = unify_edge_types_directed_undirected(graph)

    # Parse the model output
    measurement_model = model_output.get("measurement_model", [])
    structural_model = model_output.get("structural_model", [])
    residual_covariances = model_output.get("residual_covariances", [])

    # Create mappings for quick lookup
    # Measurement: (LV, Item) -> coefficient
    measurement_dict = {}
    if measurement_model:
        for mm in measurement_model:
            lv = mm["LV"]
            item = mm["Item"]
            std_estimate = mm.get("Std.Estimate", None)  # Using standardized estimate
            if std_estimate is not None:
                measurement_dict[(lv, item)] = std_estimate

    # Structural: (Predictor, LV) -> coefficient
    structural_dict = {}
    if structural_model:
        for sm in structural_model:
            lv = sm["LV"]
            predictor = sm["Predictor"]
            coef = sm["Coefficient"]
            structural_dict[(predictor, lv)] = coef

    # Residual Covariances: (Var1, Var2) -> coefficient
    residual_dict = {}
    if residual_covariances:
        for rc in residual_covariances:
            var1 = rc["Variable1"]
            var2 = rc["Variable2"]
            coef = rc["Coefficient"]
            # Ensure consistent ordering
            key = tuple(sorted([var1, var2]))
            residual_dict[key] = coef

    # Now, iterate over all edges in the graph
    edges_with_coefficients = []
    for edge in graph.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()
        endpoints = (str(edge.get_endpoint1()), str(edge.get_endpoint2()))

        coefficient = None

        # Check Measurement Model
        if endpoints == ("TAIL", "ARROW"):  # LV -> Item
            key = (node1, node2)
            if key in measurement_dict:
                coefficient = measurement_dict[key]
            elif key in structural_dict:
                coefficient = structural_dict[key]
            else:
                warnings.warn(
                    f"Edge between '{node1}' and '{node2}' not found in the model; skipping."
                )
                continue
        elif endpoints == ("ARROW", "ARROW"):  # Bidirected residual covariance
            key = tuple(sorted([node1, node2]))
            if key in residual_dict:
                coefficient = residual_dict[key]
            else:
                warnings.warn(
                    f"Residual covariance between '{node1}' and '{node2}' not found in the model; skipping."
                )
                continue
        else:
            warnings.warn(
                f"Invalid edge endpoints {endpoints} for edge '{node1}'-'{node2}'; skipping."
            )
            continue

        # Assign coefficient if found
        if coefficient is not None:
            if isinstance(edge, EdgeWithCoefficient):
                edge.coefficient = coefficient
                edges_with_coefficients.append(edge)
            else:
                # Remove edge
                graph.remove_edge(edge)
                # Add edge with coefficient
                new_edge = EdgeWithCoefficient(
                    node1=edge.node1,
                    node2=edge.node2,
                    end1=edge.endpoint1,
                    end2=edge.endpoint2,
                    coefficient=coefficient,
                )
                graph.add_edge(new_edge)
                edges_with_coefficients.append(new_edge)
        else:
            # Edge not found in the model; skip with a warning
            warnings.warn(
                f"Edge between '{node1}' and '{node2}' with endpoints {endpoints} not found in the model; skipping."
            )

    return graph, edges_with_coefficients


class EdgeWithCoefficient(Edge):
    def __init__(self, node1, node2, end1, end2, coefficient=None):
        super().__init__(node1, node2, end1, end2)
        self.coefficient = coefficient

    def __str__(self):
        return f"{super().__str__()} (Coefficient: {self.coefficient})"


# Define refined colors
POSITIVE_COLOR = "#2ca02c"  # Rich medium green
NEGATIVE_COLOR = "#d62728"  # Strong medium red


def graph_with_coefficient_to_pydot(
    G: GeneralGraph,
    edges: Optional[List[Edge]] = None,
    labels: Optional[List[str]] = None,
    title: str = "",
    dpi: float = 200,
) -> pydot.Dot:
    """
    Convert a GeneralGraph object to a DOT object with edge coefficients and color coding.

    Parameters
    ----------
    G : GeneralGraph
        A graph object from causal-learn.
    edges : list, optional (default=None)
        Specific edges to include. If None, all edges from G are used.
    labels : list, optional (default=None)
        Labels for the nodes. If None, node names are used.
    title : str, optional (default="")
        The title of the graph.
    dpi : float, optional (default=200)
        The resolution of the graph.

    Returns
    -------
    pydot_g : pydot.Dot
        The DOT representation of the graph.
    """
    import pydot

    nodes = G.get_nodes()
    if labels is not None:
        assert len(labels) == len(nodes), "Length of labels must match number of nodes."

    # Initialize the pydot graph
    pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=18)
    pydot_g.obj_dict["attributes"]["dpi"] = dpi

    # Add nodes to the pydot graph
    for i, node in enumerate(nodes):
        node_label = labels[i] if labels is not None else node.get_name()
        node_shape = "square" if node.get_node_type() == NodeType.LATENT else "ellipse"
        pydot_node = pydot.Node(str(i), label=node_label, shape=node_shape)
        pydot_g.add_node(pydot_node)

    # Define a helper function to map Endpoint to pydot arrow types
    def get_g_arrow_type(endpoint: Endpoint) -> str:
        if endpoint == Endpoint.TAIL:
            return "none"
        elif endpoint == Endpoint.ARROW:
            return "normal"
        elif endpoint == Endpoint.CIRCLE:
            return "odot"
        else:
            raise NotImplementedError(f"Unknown Endpoint type: {endpoint}")

    # Use all graph edges if specific edges are not provided
    if edges is None:
        edges = G.get_graph_edges()

    # Iterate over each edge to add to the pydot graph
    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        # Determine arrow styles based on endpoints
        arrowtail = get_g_arrow_type(endpoint1)
        arrowhead = get_g_arrow_type(endpoint2)

        # Create the pydot edge with directional attributes
        dot_edge = pydot.Edge(
            str(node1_id),
            str(node2_id),
            arrowtail=arrowtail,
            arrowhead=arrowhead,
            dir="both",  # Show both directions based on endpoints
        )

        # Initialize label and color
        label = ""
        color = "black"  # Default color

        # Check if the edge has a coefficient and assign label and color accordingly
        if isinstance(edge, EdgeWithCoefficient) and edge.coefficient is not None:
            coefficient = edge.coefficient
            label = f"{coefficient:.3f}"
            if coefficient > 0:
                color = POSITIVE_COLOR
            elif coefficient < 0:
                color = NEGATIVE_COLOR
            # If coefficient is zero, retain default color
            dot_edge.set_label(label)
            dot_edge.set_color(color)

        # Optional: Handle other edge properties (e.g., penwidth, style)
        # Example:
        if Edge.Property.dd in edge.properties:
            dot_edge.set_color("green3")
        if Edge.Property.nl in edge.properties:
            dot_edge.set_penwidth(2.0)

        # Add the edge to the pydot graph
        pydot_g.add_edge(dot_edge)

    return pydot_g
