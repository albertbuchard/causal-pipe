import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
causal_pipe_pkg = types.ModuleType("causal_pipe")
causal_pipe_pkg.__path__ = [os.path.join(ROOT, "causal_pipe")]
sys.modules.setdefault("causal_pipe", causal_pipe_pkg)

import pytest
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from bcsl.graph_utils import get_bidirected_edge, get_directed_edge

from causal_pipe.utilities.graph_utilities import get_neighbors_general_graph
from causal_pipe.sem.hill_climber import GraphHillClimber
from causal_pipe.utilities.model_comparison_utilities import NO_BETTER_MODEL


def build_pag():
    names = ["X", "Y", "Z", "A", "B", "C", "D", "E", "F"]
    nodes = {name: GraphNode(name) for name in names}
    graph = GeneralGraph(list(nodes.values()))

    graph.add_edge(get_bidirected_edge(nodes["X"], nodes["Y"]))
    graph.add_edge(get_directed_edge(nodes["Y"], nodes["Z"]))
    graph.add_edge(Edge(nodes["A"], nodes["B"], Endpoint.CIRCLE, Endpoint.CIRCLE))
    graph.add_edge(Edge(nodes["C"], nodes["D"], Endpoint.CIRCLE, Endpoint.ARROW))
    graph.add_edge(Edge(nodes["E"], nodes["F"], Endpoint.TAIL, Endpoint.CIRCLE))

    return graph, nodes


def dummy_score(graph, compared_to_graph=None):
    if compared_to_graph is None:
        return {"score": 0}
    return {"score": 0, "is_better_model": NO_BETTER_MODEL}


def test_pag_neighbor_generation_respects_pag():
    graph, nodes = build_pag()
    neighbors, _ = get_neighbors_general_graph(graph, respect_pag=True)
    assert len(neighbors) == 4

    expected_moves = {("A", "B"), ("B", "A"), ("C", "D"), ("E", "F")}
    actual_moves = set()
    for g in neighbors:
        for src, dst in expected_moves:
            e = g.get_edge(nodes[src], nodes[dst])
            if e and e.endpoint1 == Endpoint.TAIL and e.endpoint2 == Endpoint.ARROW:
                actual_moves.add((src, dst))
        e_xy = g.get_edge(nodes["X"], nodes["Y"])
        assert e_xy.endpoint1 == Endpoint.ARROW and e_xy.endpoint2 == Endpoint.ARROW
        e_yz = g.get_edge(nodes["Y"], nodes["Z"])
        assert e_yz.endpoint1 == Endpoint.TAIL and e_yz.endpoint2 == Endpoint.ARROW

    assert actual_moves == expected_moves


def test_hill_climber_preserves_circles_when_respecting_pag():
    a = GraphNode("A")
    b = GraphNode("B")
    graph = GeneralGraph(nodes=[a, b])
    graph.add_edge(Edge(a, b, Endpoint.CIRCLE, Endpoint.CIRCLE))

    climber = GraphHillClimber(
        score_function=dummy_score,
        get_neighbors_func=get_neighbors_general_graph,
        node_names=["A", "B"],
        respect_pag=True,
    )
    result = climber.run(graph, max_iter=1)
    edge = result.get_edge(a, b)
    assert edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.CIRCLE


def test_hill_climber_unifies_circles_by_default():
    a = GraphNode("A")
    b = GraphNode("B")
    graph = GeneralGraph(nodes=[a, b])
    graph.add_edge(Edge(a, b, Endpoint.CIRCLE, Endpoint.CIRCLE))

    climber = GraphHillClimber(
        score_function=dummy_score,
        get_neighbors_func=get_neighbors_general_graph,
        node_names=["A", "B"],
    )
    result = climber.run(graph, max_iter=1)
    edge = result.get_edge(a, b)
    assert edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.ARROW
