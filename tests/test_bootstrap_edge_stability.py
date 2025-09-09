import os
import sys
import types
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
causal_pipe_pkg = types.ModuleType("causal_pipe")
causal_pipe_pkg.__path__ = [os.path.join(ROOT, "causal_pipe")]
sys.modules.setdefault("causal_pipe", causal_pipe_pkg)

# Stub out heavy dependencies from causallearn and static visualization
causallearn = types.ModuleType("causallearn")
causallearn_utils = types.ModuleType("causallearn.utils")
causallearn_utils.__path__ = []
causallearn_utils_cit = types.ModuleType("causallearn.utils.cit")
causallearn_utils_FAS = types.ModuleType("causallearn.utils.FAS")
causallearn_graph = types.ModuleType("causallearn.graph")
causallearn_graph.__path__ = []
causallearn_graph_GeneralGraph = types.ModuleType("causallearn.graph.GeneralGraph")
causallearn_graph_GraphNode = types.ModuleType("causallearn.graph.GraphNode")
causallearn_graph_Edge = types.ModuleType("causallearn.graph.Edge")
causallearn_graph_Endpoint = types.ModuleType("causallearn.graph.Endpoint")
causallearn_graph_NodeType = types.ModuleType("causallearn.graph.NodeType")


class DummyCIT:
    def __init__(self, *args, **kwargs):
        pass


def dummy_fas(*args, **kwargs):
    nodes = kwargs.get("nodes") or []

    class MockEdge:
        def __init__(self, n1, n2):
            self._n1, self._n2 = n1, n2

        def get_node1(self):
            return self._n1

        def get_node2(self):
            return self._n2

    class MockGraph:
        def __init__(self, edges):
            self._edges = edges

        def get_graph_edges(self):
            return self._edges

    if len(nodes) >= 2:
        g = MockGraph([MockEdge(nodes[0], nodes[1])])
    else:
        g = MockGraph([])
    return g, {(0, 1): {2}}, None


class GraphNode:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


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


class Edge:
    def __init__(self, n1, n2, e1=None, e2=None):
        self.node1, self.node2 = n1, n2
        self.endpoint1 = e1
        self.endpoint2 = e2

    def get_node1(self):
        return self.node1

    def get_node2(self):
        return self.node2


class Endpoint(dict):
    def __getattr__(self, name):
        return self[name]

 
Endpoint = Endpoint(TAIL="TAIL", ARROW="ARROW", CIRCLE="CIRCLE") 

causallearn_utils_cit.CIT = DummyCIT
causallearn_utils_FAS.fas = dummy_fas
causallearn_graph_GeneralGraph.GeneralGraph = GeneralGraph
causallearn_graph_GraphNode.GraphNode = GraphNode
causallearn_graph_Edge.Edge = Edge
causallearn_graph_Endpoint.Endpoint = Endpoint
causallearn_graph_NodeType.NodeType = type("NodeType", (), {})

bcsl_graph_utils = types.ModuleType("bcsl.graph_utils")
def _undirected(n1, n2):
    return Edge(n1, n2, Endpoint.CIRCLE, Endpoint.CIRCLE)


def _directed(n1, n2):
    return Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)


def _bidirected(n1, n2):
    return Edge(n1, n2, Endpoint.ARROW, Endpoint.ARROW)


bcsl_graph_utils.get_nondirected_edge = _undirected
bcsl_graph_utils.get_undirected_edge = _undirected
bcsl_graph_utils.get_directed_edge = _directed
bcsl_graph_utils.get_bidirected_edge = _bidirected
pydot = types.ModuleType("pydot")
pydot.Dot = type("Dot", (), {})
pydot.Node = type("Node", (), {})
pydot.Edge = type("Edge", (), {})

sys.modules.setdefault("causallearn", causallearn)
sys.modules.setdefault("causallearn.utils", causallearn_utils)
sys.modules.setdefault("causallearn.utils.cit", causallearn_utils_cit)
sys.modules.setdefault("causallearn.utils.FAS", causallearn_utils_FAS)
sys.modules.setdefault("causallearn.graph", causallearn_graph)
sys.modules.setdefault("causallearn.graph.GeneralGraph", causallearn_graph_GeneralGraph)
sys.modules.setdefault("causallearn.graph.GraphNode", causallearn_graph_GraphNode)
sys.modules.setdefault("causallearn.graph.Edge", causallearn_graph_Edge)
sys.modules.setdefault("causallearn.graph.Endpoint", causallearn_graph_Endpoint)
sys.modules.setdefault("causallearn.graph.NodeType", causallearn_graph_NodeType)
sys.modules.setdefault("bcsl.graph_utils", bcsl_graph_utils)
sys.modules.setdefault("pydot", pydot)

static_causal_discovery = types.ModuleType(
    "causal_pipe.causal_discovery.static_causal_discovery"
)
static_causal_discovery.visualize_graph = lambda *args, **kwargs: None
sys.modules.setdefault(
    "causal_pipe.causal_discovery.static_causal_discovery", static_causal_discovery
)

from causal_pipe.causal_discovery.fas_bootstrap import (
    bootstrap_fas_edge_stability,
)

def test_bootstrap_fas_edge_stability_returns_probabilities():
    np.random.seed(0)
    n = 100
    a = np.random.randn(n)
    b = a + np.random.randn(n) * 0.1
    c = b + np.random.randn(n) * 0.1
    data = pd.DataFrame({"A": a, "B": b, "C": c})

    probs, best_graph = bootstrap_fas_edge_stability(
        data, resamples=2, random_state=1
    )
    assert isinstance(probs, dict)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert best_graph is None or isinstance(best_graph, tuple)
    if best_graph is not None:
        _, _, _, sepsets = best_graph
        assert all(isinstance(k, tuple) and len(k) == 2 for k in sepsets)
        assert all(isinstance(i, int) for k in sepsets for i in k)
        assert all(isinstance(v, set) for v in sepsets.values())
        assert all(isinstance(i, int) for v in sepsets.values() for i in v)


def test_fas_bootstrap_saves_graph_with_highest_edge_probability_product(monkeypatch, tmp_path):
    data = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]})

    class MockNode:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class MockEdge:
        def __init__(self, n1, n2):
            self._n1 = n1
            self._n2 = n2

        def get_node1(self):
            return self._n1

        def get_node2(self):
            return self._n2

    class MockGraph:
        def __init__(self, edges):
            self._edges = edges

        def get_graph_edges(self):
            return self._edges

    A, B, C = MockNode("A"), MockNode("B"), MockNode("C")
    g1 = MockGraph([MockEdge(A, B)])
    g2 = MockGraph([MockEdge(A, B), MockEdge(B, C)])

    graphs = iter([g2, g2, g1])

    def fas_mock(*args, **kwargs):
        return next(graphs), {}, None

    monkeypatch.setattr("causal_pipe.causal_discovery.fas_bootstrap.fas", fas_mock)

    class DummyCIT:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("causal_pipe.causal_discovery.fas_bootstrap.CIT", DummyCIT)

    captured = []

    def viz_mock(graph_obj, title, show, output_path):
        captured.append((graph_obj, title))

    monkeypatch.setattr(
        "causal_pipe.causal_discovery.fas_bootstrap.visualize_graph", viz_mock
    )

    bootstrap_fas_edge_stability(
        data, resamples=3, random_state=0, output_dir=str(tmp_path)
    )

    assert len(captured) == 2
    first_graph, first_title = captured[0]
    second_graph, second_title = captured[1]

    assert len(first_graph.get_graph_edges()) == 2
    assert "p=0.67" in first_title
    assert len(second_graph.get_graph_edges()) == 1
    assert "p=0.33" in second_title

