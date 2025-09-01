# causal_pipe/graph_hill_climber.py

import warnings
from typing import List, Tuple, Optional, Protocol, Set, Dict, Any

from causallearn.graph.Edge import Edge
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.utilities.graph_utilities import (
    get_all_directed_edges_list,
    unify_edge_types_directed_undirected,
    switch_directed_edge_in_graph,
    make_edge_undirected,
    is_edge_directed,
)
from causal_pipe.utilities.model_comparison_utilities import (
    NO_BETTER_MODEL,
    BETTER_MODEL_1,
)


class ScoreFunction(Protocol):
    """
    Protocol for a scoring function that evaluates and compares two graph models.
    """

    def __call__(
        self, model_1: GeneralGraph, model_2: Optional[GeneralGraph] = None
    ) -> Dict[str, Any]:
        """
        Compare two graph models and return a dictionary with scoring metrics.

        :param model_1: The primary graph model to evaluate.
        :param model_2: An optional secondary graph model for comparison.
        :return: A dictionary containing scoring information, e.g., {'score': float, 'is_better_model': int}.
        """
        ...


class GetNeighborsFunction(Protocol):
    """
    Protocol for a function that generates neighboring graphs by modifying edges.
    """

    def __call__(
        self,
        graph: GeneralGraph,
        undirected_edges: Set[Tuple[int, int]],
        kept_edges: Set[Tuple[int, int]],
        *,
        respect_pag: bool,
    ) -> Tuple[List[GeneralGraph], List[Edge]]:
        """
        Generate neighboring graphs by altering the specified edges.

        :param graph: The current graph from which neighbors are generated.
        :param undirected_edges: A set of undirected edge tuples to consider for modifications.
        :param kept_edges: A set of edges that should remain unchanged during neighbor generation.
        :return: A tuple containing a list of neighboring graphs and the corresponding list of edges that were switched.
        """
        ...


class GraphHillClimber:
    """
    A class that performs hill-climbing search on graph structures to optimize their configurations
    based on a provided scoring function.
    """

    def __init__(
        self,
        score_function: ScoreFunction,
        get_neighbors_func: GetNeighborsFunction,
        node_names: List[str],
        keep_initially_oriented_edges: bool = True,
        respect_pag: bool = False,
    ):
        """
        Initialize the HillClimber.

        :param score_function: A function to evaluate and compare graph models.
        :param get_neighbors_func: A function to generate neighboring graphs from the current graph.
        :param node_names: A list of node names present in the graph.
        :param keep_initially_oriented_edges: If True, initially directed edges are preserved during the search.
        :param respect_pag: When True, the search preserves PAG marks (circles) during optimization.
        """
        self.score_function = score_function
        self.get_neighbors_func = get_neighbors_func
        self.node_names = node_names
        self.keep_initially_oriented_edges = keep_initially_oriented_edges
        self.respect_pag = respect_pag

    def run(self, initial_graph: GeneralGraph, max_iter: int = 1000) -> GeneralGraph:
        """
        Perform hill-climbing search to find the optimal graph configuration by iteratively improving the graph's score.

        This method optimizes the edge orientations in the graph to maximize the scoring function.

        :param initial_graph: The starting graph for the hill-climbing algorithm (typically the global skeleton).
        :param max_iter: The maximum number of iterations to perform during the search.
        :return: The optimized graph after hill-climbing.
        :raises ValueError: If the score function does not return the required keys.
        """

        # Initialize the current graph with the initial graph provided
        current_graph = initial_graph
        nodes_map = current_graph.node_map  # Mapping from node indices to node names

        # Initialize kept_edges if preserving initially oriented edges is desired
        kept_edges: Optional[List[Tuple[int, int]]] = None
        if self.keep_initially_oriented_edges:
            kept_edges = get_all_directed_edges_list(current_graph)

        if not self.respect_pag:
            # Convert all edge types to undirected for uniform processing
            current_graph = unify_edge_types_directed_undirected(current_graph)

        # Evaluate the score of the initial graph
        current_score_fn_output = self.score_function(current_graph)
        if not isinstance(current_score_fn_output, dict):
            raise ValueError(
                "The score function must return a dictionary with 'score' key."
            )
        current_score = current_score_fn_output["score"]

        # Initialize iteration counter
        iteration = 0
        print(f"[SEM-Climb] respect_pag={self.respect_pag}")
        print(f"Hill Climbing started with a maximum of {max_iter} iterations.")
        print(f"Initial score = {current_score}")

        # Initialize list to track undirected edges that should not be modified further
        undirected_edges: List[Tuple[int, int]] = []

        # Begin hill-climbing iterations
        while iteration < max_iter:
            # Provide periodic updates every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best score = {current_score}")
                print(f"Current graph: {current_graph}")

            # Generate neighboring graphs and the edges that were switched to obtain them
            neighbors, switched_edges = self.get_neighbors_func(
                current_graph,
                undirected_edges=undirected_edges,
                kept_edges=set(kept_edges) if kept_edges else set(),
                respect_pag=self.respect_pag,
            )

            # Initialize variables to track the best neighbor in this iteration
            best_neighbor: Optional[GeneralGraph] = None
            best_score = current_score
            # We reset the undirected edges for each best model
            # This is because the undirected edges are only relevant for the current graph
            # Fit equivalence might be absent in the next iteration
            undirected_edges = []

            # Iterate through each neighbor to evaluate and compare scores
            for idx, neighbor in enumerate(neighbors):
                # Evaluate the score of the neighbor graph, optionally comparing with the current graph
                neighbor_score_fn_output = self.score_function(neighbor, current_graph)

                if not isinstance(neighbor_score_fn_output, dict):
                    raise ValueError(
                        "The score function must return a dictionary with 'score' key."
                    )

                # Determine if the neighbor is a better model based on the scoring function
                is_better_model = neighbor_score_fn_output.get("is_better_model")

                if is_better_model is not None:
                    if is_better_model == BETTER_MODEL_1:
                        # If the neighbor model is better, check if its score is the best so far
                        if neighbor_score_fn_output["score"] > best_score:
                            best_score = neighbor_score_fn_output["score"]
                            best_neighbor = neighbor
                    elif is_better_model == NO_BETTER_MODEL:
                        # Handle cases where the neighbor model is not directly better but might allow further exploration
                        changed_edge: Edge = switched_edges[idx]
                        edge_in_neighbor = neighbor.get_edge(
                            changed_edge.node1, changed_edge.node2
                        )

                        if edge_in_neighbor is None:
                            raise ValueError(
                                "The edge should be present in the neighbor graph."
                            )

                        # Flags to determine the type of edge modification required
                        should_make_undirected = False
                        should_switch_current = False
                        should_switch_neighbour = False

                        is_better_model_undirected: Optional[int] = None

                        # Determine the directionality of the edge in both current and neighbor graphs
                        if is_edge_directed(changed_edge):
                            if is_edge_directed(edge_in_neighbor):
                                should_make_undirected = True
                            else:
                                should_switch_current = True
                        else:
                            if is_edge_directed(edge_in_neighbor):
                                should_switch_neighbour = True
                            else:
                                raise ValueError(
                                    "The edge should be directed in the neighbor graph."
                                )

                        # Handle making the edge undirected if applicable
                        if should_make_undirected:
                            neighbor_undirected_edge = make_edge_undirected(
                                neighbor, changed_edge
                            )
                            neighbor_undirected_score_fn_output = self.score_function(
                                neighbor_undirected_edge, neighbor
                            )
                            is_better_model_undirected = (
                                neighbor_undirected_score_fn_output.get(
                                    "is_better_model"
                                )
                            )
                        # Handle switching the direction of the edge
                        elif should_switch_current or should_switch_neighbour:
                            if should_switch_current:
                                neighbor_switched_edge = switch_directed_edge_in_graph(
                                    current_graph, changed_edge
                                )
                            else:
                                neighbor_switched_edge = switch_directed_edge_in_graph(
                                    neighbor, edge_in_neighbor
                                )
                            neighbor_switched_score_fn_output = self.score_function(
                                neighbor_switched_edge, neighbor
                            )
                            is_better_model_undirected = (
                                neighbor_switched_score_fn_output.get("is_better_model")
                            )

                        # Ensure the score function provided the necessary key
                        if is_better_model_undirected is None:
                            warnings.warn(
                                "The score function must return a dictionary with 'is_better_model' key."
                            )

                        # If the undirected or switched edge leads to no better model, track it to avoid future modifications
                        if is_better_model_undirected == NO_BETTER_MODEL:
                            edge_node_idx = (
                                nodes_map[changed_edge.node1],
                                nodes_map[changed_edge.node2],
                            )
                            undirected_edges.append(edge_node_idx)
                else:
                    raise ValueError(
                        "The score function must return a dictionary with 'is_better_model' key."
                    )

            # If no better neighbor is found, terminate the hill-climbing process
            if best_neighbor is None:
                print(
                    f"Iteration {iteration}: No better neighbor found. Stopping. Best score = {current_score}"
                )
                break

            # Update the current graph and its score with the best neighbor found
            current_graph = best_neighbor
            current_score = best_score

            # Convert undirected_edges to a set for efficient look-up in the next iteration
            undirected_edges = set(undirected_edges)

            # Increment the iteration counter
            iteration += 1

            # Notify if the maximum number of iterations is reached without convergence
            if iteration == max_iter:
                print(
                    f"Max iteration reached without convergence. Best score = {current_score}"
                )

        optimized_graph = (
            current_graph
            if self.respect_pag
            else unify_edge_types_directed_undirected(current_graph)
        )

        return optimized_graph
