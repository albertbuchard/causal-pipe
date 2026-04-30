import os
import sys
import types

import pandas as pd

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

causallearn = types.ModuleType("causallearn")
causallearn_utils = types.ModuleType("causallearn.utils")
causallearn_utils_cit = types.ModuleType("causallearn.utils.cit")
causallearn_utils_FAS = types.ModuleType("causallearn.utils.FAS")
causallearn_graph = types.ModuleType("causallearn.graph")
causallearn_graph_GeneralGraph = types.ModuleType("causallearn.graph.GeneralGraph")
causallearn_graph_GraphNode = types.ModuleType("causallearn.graph.GraphNode")
causallearn_graph_Edge = types.ModuleType("causallearn.graph.Edge")
causallearn_graph_Endpoint = types.ModuleType("causallearn.graph.Endpoint")
causallearn_graph_NodeType = types.ModuleType("causallearn.graph.NodeType")
bcsl_graph_utils = types.ModuleType("bcsl.graph_utils")
pydot = types.ModuleType("pydot")


class GraphNode:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class Edge:
    def __init__(self, n1=None, n2=None, e1=None, e2=None):
        self.node1, self.node2 = n1, n2
        self.endpoint1 = e1
        self.endpoint2 = e2


class Endpoint(dict):
    def __getattr__(self, name):
        return self[name]


class GeneralGraph:
    def __init__(self, nodes):
        self._nodes = nodes
        self._edges = []

    def add_edge(self, edge):
        self._edges.append(edge)

    def get_graph_edges(self):
        return self._edges


Endpoint = Endpoint(TAIL="TAIL", ARROW="ARROW", CIRCLE="CIRCLE")
causallearn_utils_cit.CIT = lambda *args, **kwargs: None
causallearn_utils_FAS.fas = lambda *args, **kwargs: (GeneralGraph([]), {}, None)
causallearn_graph_GeneralGraph.GeneralGraph = GeneralGraph
causallearn_graph_GraphNode.GraphNode = GraphNode
causallearn_graph_Edge.Edge = Edge
causallearn_graph_Endpoint.Endpoint = Endpoint
causallearn_graph_NodeType.NodeType = type("NodeType", (), {})
bcsl_graph_utils.get_nondirected_edge = lambda n1, n2: Edge(n1, n2, Endpoint.TAIL, Endpoint.TAIL)
bcsl_graph_utils.get_undirected_edge = lambda n1, n2: Edge(n1, n2, Endpoint.TAIL, Endpoint.TAIL)
bcsl_graph_utils.get_directed_edge = lambda n1, n2: Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)
bcsl_graph_utils.get_bidirected_edge = lambda n1, n2: Edge(n1, n2, Endpoint.ARROW, Endpoint.ARROW)
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

from causal_pipe.causal_discovery import fas_bootstrap


def test_cluster_bootstrap_resamples_whole_clusters(monkeypatch):
    data = pd.DataFrame({"x": [1, 2, 10, 20], "y": [2, 3, 11, 21]})
    labels = ["a", "a", "b", "b"]
    seen_samples = []

    def fas_mock(data, nodes, independence_test_method, **kwargs):
        seen_samples.append(pd.DataFrame(data, columns=["x", "y"]))
        return GeneralGraph([]), {}, None

    monkeypatch.setattr(fas_bootstrap, "fas", fas_mock)

    fas_bootstrap.bootstrap_fas_edge_stability(
        data,
        resamples=2,
        random_state=1,
        bootstrap_unit="cluster",
        cluster_labels=labels,
    )

    assert seen_samples
    for sample in seen_samples:
        assert len(sample) == 4
        sampled_pairs = [
            tuple(map(tuple, sample.iloc[i : i + 2].to_numpy()))
            for i in range(0, len(sample), 2)
        ]
        assert all(
            pair
            in [
                ((1.0, 2.0), (2.0, 3.0)),
                ((10.0, 11.0), (20.0, 21.0)),
            ]
            for pair in sampled_pairs
        )


def test_block_bootstrap_preserves_contiguous_windows(monkeypatch):
    data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    seen_samples = []

    def fas_mock(data, nodes, independence_test_method, **kwargs):
        seen_samples.append(pd.DataFrame(data, columns=["x", "y"]))
        return GeneralGraph([]), {}, None

    monkeypatch.setattr(fas_bootstrap, "fas", fas_mock)

    fas_bootstrap.bootstrap_fas_edge_stability(
        data,
        resamples=1,
        random_state=2,
        bootstrap_unit="block",
        block_length=2,
    )

    sample = seen_samples[0]
    assert len(sample) == 4
    for i in range(0, len(sample), 2):
        block = sample.iloc[i : i + 2]["x"].tolist()
        assert block[1] - block[0] == 1
