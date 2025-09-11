import os
import sys
import types
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
causal_pipe_pkg = types.ModuleType("causal_pipe")
causal_pipe_pkg.__path__ = [os.path.join(ROOT, "causal_pipe")]
sys.modules.setdefault("causal_pipe", causal_pipe_pkg)

import importlib

for mod in [
    "causallearn",
    "causallearn.graph",
    "causallearn.graph.GraphNode",
    "causallearn.graph.GeneralGraph",
    "causallearn.graph.Edge",
    "causallearn.graph.Endpoint",
    "causallearn.graph.NodeType",
    "causallearn.utils",
    "causallearn.utils.cit",
    "causallearn.utils.GraphUtils",
    "bcsl.graph_utils",
]:
    sys.modules.pop(mod, None)

bcsl_graph_utils = types.ModuleType("bcsl.graph_utils")
bcsl_graph_utils.get_nondirected_edge = (
    lambda n1, n2: Edge(n1, n2, Endpoint.TAIL, Endpoint.TAIL)
)
bcsl_graph_utils.get_undirected_edge = (
    lambda n1, n2: Edge(n1, n2, Endpoint.TAIL, Endpoint.TAIL)
)
bcsl_graph_utils.get_directed_edge = (
    lambda n1, n2: Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)
)
bcsl_graph_utils.get_bidirected_edge = (
    lambda n1, n2: Edge(n1, n2, Endpoint.ARROW, Endpoint.ARROW)
)
sys.modules["bcsl.graph_utils"] = bcsl_graph_utils

import types

# Stub out heavy dependencies from causallearn and bcsl.graph_utils
causallearn = types.ModuleType("causallearn")
causallearn_graph = types.ModuleType("causallearn.graph")
causallearn_graph_GraphNode = types.ModuleType("causallearn.graph.GraphNode")
causallearn_graph_GeneralGraph = types.ModuleType("causallearn.graph.GeneralGraph")
causallearn_graph_Edge = types.ModuleType("causallearn.graph.Edge")
causallearn_graph_Endpoint = types.ModuleType("causallearn.graph.Endpoint")
causallearn_graph_NodeType = types.ModuleType("causallearn.graph.NodeType")


class GraphNode:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class Edge:
    def __init__(self, n1, n2, e1=None, e2=None):
        self.node1, self.node2 = n1, n2
        self.endpoint1, self.endpoint2 = e1, e2

    def get_endpoint(self, node):
        if node == self.node1:
            return self.endpoint1
        if node == self.node2:
            return self.endpoint2
        return None


class Endpoint(dict):
    def __getattr__(self, name):
        return self[name]


Endpoint = Endpoint(TAIL="TAIL", ARROW="ARROW", CIRCLE="CIRCLE")


class GeneralGraph:
    def __init__(self, nodes):
        self._nodes = nodes
        self._edges = []
        self.node_map = {node: i for i, node in enumerate(nodes)}

    def add_edge(self, edge):
        if edge is not None:
            self._edges.append(edge)

    def add_directed_edge(self, n1, n2):
        self.add_edge(Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW))

    def remove_edge(self, edge):
        if edge in self._edges:
            self._edges.remove(edge)

    def remove_connecting_edge(self, n1, n2):
        edge = self.get_edge(n1, n2)
        if edge in self._edges:
            self._edges.remove(edge)

    def get_graph_edges(self):
        return self._edges

    def get_nodes(self):
        return self._nodes

    def get_num_nodes(self):
        return len(self._nodes)

    def get_edge(self, n1, n2):
        for e in self._edges:
            if e is None:
                continue
            if (e.node1 == n1 and e.node2 == n2) or (e.node1 == n2 and e.node2 == n1):
                return e
        return None

    def get_node_edges(self, node):
        return [e for e in self._edges if e.node1 == node or e.node2 == node]

    def is_adjacent_to(self, n1, n2):
        return self.get_edge(n1, n2) is not None

    def is_directed_from_to(self, n1, n2):
        e = self.get_edge(n1, n2)
        if e is None:
            return False
        if e.node1 == n1 and e.endpoint1 == Endpoint.TAIL and e.endpoint2 == Endpoint.ARROW:
            return True
        if e.node2 == n1 and e.endpoint2 == Endpoint.TAIL and e.endpoint1 == Endpoint.ARROW:
            return True
        return False

    def get_children(self, node):
        children = []
        for e in self._edges:
            if e.node1 == node and e.endpoint1 == Endpoint.TAIL and e.endpoint2 == Endpoint.ARROW:
                children.append(e.node2)
            elif e.node2 == node and e.endpoint2 == Endpoint.TAIL and e.endpoint1 == Endpoint.ARROW:
                children.append(e.node1)
        return children

    def get_parents(self, node):
        parents = []
        for e in self._edges:
            if e.node1 == node and e.endpoint1 == Endpoint.ARROW and e.endpoint2 == Endpoint.TAIL:
                parents.append(e.node2)
            elif e.node2 == node and e.endpoint2 == Endpoint.ARROW and e.endpoint1 == Endpoint.TAIL:
                parents.append(e.node1)
        return parents


causallearn_graph_GraphNode.GraphNode = GraphNode
causallearn_graph_GeneralGraph.GeneralGraph = GeneralGraph
causallearn_graph_Edge.Edge = Edge
causallearn_graph_Endpoint.Endpoint = Endpoint
class NodeType:
    pass
causallearn_graph_NodeType.NodeType = NodeType

sys.modules.setdefault("causallearn", causallearn)
sys.modules.setdefault("causallearn.graph", causallearn_graph)
sys.modules.setdefault("causallearn.graph.GraphNode", causallearn_graph_GraphNode)
sys.modules.setdefault("causallearn.graph.GeneralGraph", causallearn_graph_GeneralGraph)
sys.modules.setdefault("causallearn.graph.Edge", causallearn_graph_Edge)
sys.modules.setdefault("causallearn.graph.Endpoint", causallearn_graph_Endpoint)
sys.modules.setdefault("causallearn.graph.NodeType", causallearn_graph_NodeType)

bcsl_graph_utils = types.ModuleType("bcsl.graph_utils")


def get_bidirected_edge(n1, n2):
    return Edge(n1, n2, Endpoint.ARROW, Endpoint.ARROW)


def get_directed_edge(n1, n2):
    return Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)


bcsl_graph_utils.get_bidirected_edge = get_bidirected_edge
bcsl_graph_utils.get_directed_edge = get_directed_edge
sys.modules.setdefault("bcsl.graph_utils", bcsl_graph_utils)

from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
import causal_pipe.utilities.graph_utilities as graph_utils
importlib.reload(graph_utils)
from causal_pipe.utilities.graph_utilities import (
    get_neighbors_general_graph,
    general_graph_to_sem_model,
)
import causal_pipe.hill_climber.hill_climber as hill_climber
importlib.reload(hill_climber)
from causal_pipe.hill_climber.hill_climber import GraphHillClimber
from causal_pipe.utilities.model_comparison_utilities import (
    NO_BETTER_MODEL,
    BETTER_MODEL_1,
)


def get_bidirected_edge(n1, n2):
    return Edge(n1, n2, Endpoint.ARROW, Endpoint.ARROW)


def get_directed_edge(n1, n2):
    return Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)


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


def test_hill_climber_prefers_neighbor_on_circle_tie():
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

    assert edge.endpoint1 != Endpoint.CIRCLE and edge.endpoint2 != Endpoint.CIRCLE

    neighbors, _ = get_neighbors_general_graph(graph, respect_pag=True)
    neighbor_endpoints = [
        (n.get_edge(a, b).endpoint1, n.get_edge(a, b).endpoint2) for n in neighbors
    ]
    assert (edge.endpoint1, edge.endpoint2) in neighbor_endpoints


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


def test_hill_climber_better_model_wins_on_circle_edge():
    a = GraphNode("A")
    b = GraphNode("B")
    graph = GeneralGraph(nodes=[a, b])
    graph.add_edge(Edge(a, b, Endpoint.CIRCLE, Endpoint.CIRCLE))

    def favor_a_to_b(g: GeneralGraph, compared_to_graph=None):
        if compared_to_graph is None:
            return {"score": 0}
        e = g.get_edge(a, b)
        if e.endpoint1 == Endpoint.TAIL and e.endpoint2 == Endpoint.ARROW:
            return {"score": 1, "is_better_model": BETTER_MODEL_1}
        return {"score": 0, "is_better_model": BETTER_MODEL_1}

    climber = GraphHillClimber(
        score_function=favor_a_to_b,
        get_neighbors_func=get_neighbors_general_graph,
        node_names=["A", "B"],
        respect_pag=True,
    )
    result = climber.run(graph, max_iter=1)
    edge = result.get_edge(a, b)
    assert edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW


def test_general_graph_to_sem_model_errors_on_circle():
    a = GraphNode("A")
    b = GraphNode("B")
    graph = GeneralGraph(nodes=[a, b])
    graph.add_edge(Edge(a, b, Endpoint.CIRCLE, Endpoint.CIRCLE))

    with pytest.raises(ValueError):
        general_graph_to_sem_model(graph, on_circle="error")
