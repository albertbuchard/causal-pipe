import os
import sys
import types
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
causal_pipe_pkg = types.ModuleType("causal_pipe")
causal_pipe_pkg.__path__ = [os.path.join(ROOT, "causal_pipe")]
sys.modules.setdefault("causal_pipe", causal_pipe_pkg)

causal_pipe_cd_pkg = types.ModuleType("causal_pipe.causal_discovery")
causal_pipe_cd_pkg.__path__ = [os.path.join(ROOT, "causal_pipe", "causal_discovery")]
sys.modules.setdefault("causal_pipe.causal_discovery", causal_pipe_cd_pkg)
static_causal_discovery = types.ModuleType(
    "causal_pipe.causal_discovery.static_causal_discovery"
)
static_causal_discovery.visualize_graph = lambda *args, **kwargs: None
sys.modules.setdefault(
    "causal_pipe.causal_discovery.static_causal_discovery", static_causal_discovery
)
setattr(causal_pipe_pkg, "causal_discovery", causal_pipe_cd_pkg)
setattr(causal_pipe_cd_pkg, "static_causal_discovery", static_causal_discovery)

# Create minimal stub modules for causallearn
causallearn = types.ModuleType("causallearn")
causallearn_utils = types.ModuleType("causallearn.utils")
causallearn_utils_cit = types.ModuleType("causallearn.utils.cit")
causallearn_utils_FAS = types.ModuleType("causallearn.utils.FAS")
causallearn_graph = types.ModuleType("causallearn.graph")
causallearn_graph_GeneralGraph = types.ModuleType("causallearn.graph.GeneralGraph")
causallearn_graph_Edge = types.ModuleType("causallearn.graph.Edge")
causallearn_graph_Endpoint = types.ModuleType("causallearn.graph.Endpoint")
causallearn_graph_GraphNode = types.ModuleType("causallearn.graph.GraphNode")
causallearn_graph_NodeType = types.ModuleType("causallearn.graph.NodeType")
bcsl_graph_utils = types.ModuleType("bcsl.graph_utils")
pydot = types.ModuleType("pydot")


class _DummyGraph:
    def __init__(self, edges):
        self._edges = edges

    def get_graph_edges(self):
        return self._edges

    def get_nodes(self):
        return []


causallearn_graph_GeneralGraph.GeneralGraph = _DummyGraph
causallearn_graph_Edge.Edge = type("Edge", (), {})
class _Endpoint(dict):
    def __getattr__(self, item):
        return self[item]


causallearn_graph_Endpoint.Endpoint = _Endpoint(
    TAIL="TAIL", ARROW="ARROW", CIRCLE="CIRCLE"
)


class GraphNode:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


causallearn_graph_GraphNode.GraphNode = GraphNode
causallearn_graph_NodeType.NodeType = type("NodeType", (), {})

bcsl_graph_utils.get_nondirected_edge = lambda *args, **kwargs: None
bcsl_graph_utils.get_undirected_edge = lambda *args, **kwargs: None
bcsl_graph_utils.get_directed_edge = lambda *args, **kwargs: None
bcsl_graph_utils.get_bidirected_edge = lambda *args, **kwargs: None
pydot.Dot = type("Dot", (), {})
pydot.Node = type("Node", (), {})
pydot.Edge = type("Edge", (), {})


class _DummyCIT:
    def __init__(self, *args, **kwargs):
        pass


causallearn_utils_cit.CIT = _DummyCIT

# placeholder fas function
causallearn_utils_FAS.fas = lambda *args, **kwargs: None

# register stub modules
sys.modules.setdefault("causallearn", causallearn)
sys.modules.setdefault("causallearn.utils", causallearn_utils)
sys.modules.setdefault("causallearn.utils.cit", causallearn_utils_cit)
sys.modules.setdefault("causallearn.utils.FAS", causallearn_utils_FAS)
sys.modules.setdefault("causallearn.graph", causallearn_graph)
sys.modules.setdefault("causallearn.graph.GeneralGraph", causallearn_graph_GeneralGraph)
sys.modules.setdefault("causallearn.graph.Edge", causallearn_graph_Edge)
sys.modules.setdefault("causallearn.graph.Endpoint", causallearn_graph_Endpoint)
sys.modules.setdefault("causallearn.graph.GraphNode", causallearn_graph_GraphNode)
sys.modules.setdefault("causallearn.graph.NodeType", causallearn_graph_NodeType)
sys.modules.setdefault("bcsl.graph_utils", bcsl_graph_utils)
sys.modules.setdefault("pydot", pydot)

from causal_pipe.causal_discovery.fas_bootstrap import bootstrap_fas_edge_stability


def test_fas_bootstrap_returns_probabilities_without_filtering(monkeypatch):
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

    class MockGraph(_DummyGraph):
        pass

    data = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]})

    A, B, C = MockNode("A"), MockNode("B"), MockNode("C")
    g1 = MockGraph([MockEdge(A, B)])
    g2 = MockGraph([MockEdge(A, B), MockEdge(B, C)])

    graphs = iter([g2, g2, g1])

    def fas_mock(*args, **kwargs):
        return next(graphs), {}, None

    monkeypatch.setattr("causal_pipe.causal_discovery.fas_bootstrap.fas", fas_mock)

    def make_graph_mock(node_names, edges_repr):
        node_map = {name: MockNode(name) for name in node_names}
        edges = [MockEdge(node_map[a], node_map[b]) for a, b, *_ in edges_repr]
        return MockGraph(edges)

    monkeypatch.setattr(
        "causal_pipe.causal_discovery.fas_bootstrap.make_graph", make_graph_mock
    )

    probs, best_graph = bootstrap_fas_edge_stability(
        data, resamples=3, random_state=0
    )

    assert probs[("A", "B")] == 1.0
    assert pytest.approx(probs[("B", "C")], 0.01) == 2 / 3
    assert best_graph is not None
    _, graph_obj, _, _ = best_graph
    edges = [
        (e.get_node1().get_name(), e.get_node2().get_name())
        for e in graph_obj.get_graph_edges()
    ]
    assert sorted(edges) == [("A", "B"), ("B", "C")]

    threshold = 0.8
    filtered = [
        (a, b)
        for (a, b) in edges
        if probs.get((a, b), probs.get((b, a), 0.0)) >= threshold
    ]
    assert filtered == [("A", "B")]
