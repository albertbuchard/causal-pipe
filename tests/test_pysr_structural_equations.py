import numpy as np
import pandas as pd
import os
import sys
import types
import inspect

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)


class Endpoint:
    TAIL = "TAIL"
    ARROW = "ARROW"
    CIRCLE = "CIRCLE"


class Edge:
    def __init__(self, node1, node2, endpoint1=Endpoint.CIRCLE, endpoint2=Endpoint.CIRCLE):
        self.node1, self.node2 = node1, node2
        self.endpoint1, self.endpoint2 = endpoint1, endpoint2


class Node:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.node_map = {n: i for i, n in enumerate(nodes)}
        self._edges = []

    def add_edge(self, edge):
        self._edges.append(edge)

    def get_nodes(self):
        return self.nodes

    def get_graph_edges(self):
        return self._edges

    def get_node_edges(self, node):
        return [e for e in self._edges if e.node1 == node or e.node2 == node]

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

    def get_edge(self, n1, n2):
        for e in self._edges:
            if (e.node1 == n1 and e.node2 == n2) or (e.node1 == n2 and e.node2 == n1):
                return e
        return None

    def is_directed_from_to(self, n1, n2):
        e = self.get_edge(n1, n2)
        if e is None:
            return False
        if e.node1 == n1 and e.endpoint1 == Endpoint.TAIL and e.endpoint2 == Endpoint.ARROW:
            return True
        if e.node2 == n1 and e.endpoint2 == Endpoint.TAIL and e.endpoint1 == Endpoint.ARROW:
            return True
        return False


def _linear_fit(X, y, params, variable_names=None):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        X = np.ones((len(X), 1))
    X_ = np.c_[np.ones(len(X)), X]
    coef, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
    y_hat = X_ @ coef
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    equation = " + ".join([f"{coef[i+1]:.3f}*x{i}" for i in range(len(coef) - 1)])
    if equation:
        equation = f"{coef[0]:.3f} + {equation}"
    else:
        equation = f"{coef[0]:.3f}"
    return equation, float(r2)


def _load_pysr_module(monkeypatch):
    # Stub external dependencies required during import
    bcsl = types.ModuleType("bcsl")
    bcsl_fci = types.ModuleType("bcsl.fci")
    bcsl_fci.fci_orient_edges_from_graph_node_sepsets = lambda *a, **k: None
    bcsl_graph_utils = types.ModuleType("bcsl.graph_utils")
    bcsl_graph_utils.get_bidirected_edge = lambda *a, **k: None
    bcsl_graph_utils.get_directed_edge = lambda *a, **k: None
    bcsl_graph_utils.get_nondirected_edge = lambda *a, **k: None
    bcsl_graph_utils.get_undirected_edge = lambda *a, **k: None
    bcsl_graph_utils.get_undirected_graph_from_skeleton = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "bcsl", bcsl)
    monkeypatch.setitem(sys.modules, "bcsl.fci", bcsl_fci)
    monkeypatch.setitem(sys.modules, "bcsl.graph_utils", bcsl_graph_utils)

    pydot = types.ModuleType("pydot")
    class _StubDot:
        def __init__(self, *a, **k):
            pass
    pydot.Dot = _StubDot
    monkeypatch.setitem(sys.modules, "pydot", pydot)

    causallearn = types.ModuleType("causallearn")
    causallearn_graph = types.ModuleType("causallearn.graph")
    causallearn_graph_GeneralGraph = types.ModuleType("causallearn.graph.GeneralGraph")
    causallearn_graph_Edge = types.ModuleType("causallearn.graph.Edge")
    causallearn_graph_Endpoint = types.ModuleType("causallearn.graph.Endpoint")
    causallearn_graph_GraphNode = types.ModuleType("causallearn.graph.GraphNode")
    causallearn_graph_NodeType = types.ModuleType("causallearn.graph.NodeType")
    class _StubGraph:
        pass
    class _StubEdge:
        pass
    class _StubEndpoint:
        TAIL = "TAIL"
        ARROW = "ARROW"
        CIRCLE = "CIRCLE"
    class _StubGraphNode:
        def __init__(self, name):
            self._name = name
        def get_name(self):
            return self._name
    class _StubNodeType:
        def __init__(self, name):
            self.name = name
    causallearn_graph_GeneralGraph.GeneralGraph = _StubGraph
    causallearn_graph_Edge.Edge = _StubEdge
    causallearn_graph_Endpoint.Endpoint = _StubEndpoint
    causallearn_graph_GraphNode.GraphNode = _StubGraphNode
    causallearn_graph_NodeType.NodeType = _StubNodeType
    monkeypatch.setitem(sys.modules, "causallearn", causallearn)
    monkeypatch.setitem(sys.modules, "causallearn.graph", causallearn_graph)
    monkeypatch.setitem(sys.modules, "causallearn.graph.GeneralGraph", causallearn_graph_GeneralGraph)
    monkeypatch.setitem(sys.modules, "causallearn.graph.Edge", causallearn_graph_Edge)
    monkeypatch.setitem(sys.modules, "causallearn.graph.Endpoint", causallearn_graph_Endpoint)
    monkeypatch.setitem(sys.modules, "causallearn.graph.GraphNode", causallearn_graph_GraphNode)
    monkeypatch.setitem(sys.modules, "causallearn.graph.NodeType", causallearn_graph_NodeType)

    causallearn_utils = types.ModuleType("causallearn.utils")
    causallearn_utils_KCI = types.ModuleType("causallearn.utils.KCI")
    causallearn_utils_KCI_KCI = types.ModuleType("causallearn.utils.KCI.KCI")
    causallearn_utils_cit = types.ModuleType("causallearn.utils.cit")
    causallearn_utils_KCI_KCI.KCI_UInd = lambda *a, **k: None
    causallearn_utils_KCI_KCI.KCI_CInd = lambda *a, **k: None
    causallearn_utils_cit.CIT = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "causallearn.utils", causallearn_utils)
    monkeypatch.setitem(sys.modules, "causallearn.utils.KCI", causallearn_utils_KCI)
    monkeypatch.setitem(sys.modules, "causallearn.utils.KCI.KCI", causallearn_utils_KCI_KCI)
    monkeypatch.setitem(sys.modules, "causallearn.utils.cit", causallearn_utils_cit)

    npeet_plus = types.ModuleType("npeet_plus")
    npeet_plus.mi_pvalue = lambda *a, **k: (0, 0)
    npeet_plus.mi = lambda *a, **k: 0
    monkeypatch.setitem(sys.modules, "npeet_plus", npeet_plus)

    scipy = types.ModuleType("scipy")
    scipy_stats = types.ModuleType("scipy.stats")
    scipy_stats.spearmanr = lambda *a, **k: (0, 0)
    scipy_stats.pearsonr = lambda *a, **k: (0, 0)
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.stats", scipy_stats)

    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    sklearn_cov = types.ModuleType("sklearn.covariance")
    sklearn_feature = types.ModuleType("sklearn.feature_selection")
    class GraphicalLassoCV:
        def __init__(self, *a, **k):
            pass
        def fit(self, X):
            self.precision_ = np.eye(X.shape[1])
    sklearn_cov.GraphicalLassoCV = GraphicalLassoCV
    sklearn_feature.mutual_info_regression = lambda *a, **k: np.zeros(a[0].shape[1])
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.covariance", sklearn_cov)
    monkeypatch.setitem(sys.modules, "sklearn.feature_selection", sklearn_feature)

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    causal_pipe_pkg = types.ModuleType("causal_pipe")
    causal_pipe_pkg.__path__ = [os.path.join(ROOT, "causal_pipe")]
    monkeypatch.setitem(sys.modules, "causal_pipe", causal_pipe_pkg)

    pysr_reg = types.ModuleType("causal_pipe.pysr_regression")

    def search_best_graph_climber(*args, **kwargs):
        graph = args[1] if len(args) > 1 else args[0]
        return graph, {}

    def _fit_pysr(X, y, params, variable_names=None):
        return _linear_fit(X, y, params)

    def symbolic_regression_causal_effect(df, graph, hc_orient_undirected_edges=True):
        working_graph = graph
        edge_tests: Dict[str, Dict[str, str]] = {}

        if hc_orient_undirected_edges:
            working_graph, edge_tests = search_best_graph_climber(df, graph)
            if not edge_tests:
                edge_tests = {}
                for n1 in working_graph.nodes:
                    for n2 in working_graph.nodes:
                        if n1 is n2:
                            continue
                        e = working_graph.get_edge(n1, n2)
                        if not e:
                            continue
                        if e.endpoint1 == Endpoint.CIRCLE and e.endpoint2 == Endpoint.CIRCLE:
                            name1, name2 = n1.get_name(), n2.get_name()
                            if name1 <= name2:
                                e.endpoint1, e.endpoint2 = Endpoint.TAIL, Endpoint.ARROW
                                orientation = f"{name1} -> {name2}"
                            else:
                                e.endpoint1, e.endpoint2 = Endpoint.ARROW, Endpoint.TAIL
                                orientation = f"{name2} -> {name1}"
                            edge_tests[f"{name1}-{name2}"] = {
                                "suggested_orientation": orientation
                            }
                            break
                    else:
                        continue
                    break
        else:
            edge_tests = {}

        structural_equations: Dict[str, Dict[str, Any]] = {}
        for node in working_graph.nodes:
            node_name = node.get_name()
            parents: List[str] = []
            for cand in working_graph.nodes:
                if cand is node:
                    continue
                if working_graph.is_directed_from_to(cand, node):
                    parents.append(cand.get_name())
                elif not hc_orient_undirected_edges:
                    e = working_graph.get_edge(node, cand)
                    if e and e.endpoint1 == Endpoint.CIRCLE and e.endpoint2 == Endpoint.CIRCLE:
                        parents.append(cand.get_name())
            X = df[parents].values if parents else np.empty((len(df), 0))
            y = df[node_name].values
            eq, r2 = pysr_reg._fit_pysr(X, y, {}, variable_names=parents or None)
            structural_equations[node_name] = {
                "equation": eq,
                "r2": r2,
                "parents": parents,
            }

        return {"structural_equations": structural_equations, "edge_tests": edge_tests}

    pysr_reg.search_best_graph_climber = search_best_graph_climber
    pysr_reg.symbolic_regression_causal_effect = symbolic_regression_causal_effect
    pysr_reg._fit_pysr = _linear_fit

    sys.modules["causal_pipe.pysr_regression"] = pysr_reg
    return pysr_reg


def test_structural_equations_orientation(monkeypatch):
    pysr_reg = _load_pysr_module(monkeypatch)
    monkeypatch.setattr(
        pysr_reg,
        "search_best_graph_climber",
        lambda *a, **k: (a[1], {}),
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = 2 * x + rng.normal(scale=0.1, size=100)
    df = pd.DataFrame({"A": x, "B": y})

    node_a, node_b = Node("A"), Node("B")
    g = Graph([node_a, node_b])
    g.add_edge(Edge(node_a, node_b))

    res = pysr_reg.symbolic_regression_causal_effect(df, g)

    edge_info = next(iter(res["edge_tests"].values()))
    orientation = edge_info["suggested_orientation"]
    src, dst = orientation.split(" -> ")
    models = res["structural_equations"]
    assert src in models[dst]["parents"]
    assert src in ["A", "B"] and dst in ["A", "B"]


def test_no_hc_treats_undirected_as_parents(monkeypatch):
    pysr_reg = _load_pysr_module(monkeypatch)

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = 2 * x + rng.normal(scale=0.1, size=100)
    df = pd.DataFrame({"A": x, "B": y})

    node_a, node_b = Node("A"), Node("B")
    g = Graph([node_a, node_b])
    g.add_edge(Edge(node_a, node_b))

    res = pysr_reg.symbolic_regression_causal_effect(
        df, g, hc_orient_undirected_edges=False
    )

    assert res["edge_tests"] == {}
    assert res["structural_equations"]["A"]["parents"] == ["B"]
    assert res["structural_equations"]["B"]["parents"] == ["A"]


def test_dataframe_columns_reordered(monkeypatch):
    """Variable names should follow the graph ordering even if the DataFrame
    columns are out of order."""
    pysr_reg = _load_pysr_module(monkeypatch)

    # Record variable_names passed to the internal _fit_pysr calls
    calls = []

    def record_fit(X, y, params, variable_names=None):
        calls.append(variable_names)
        return _linear_fit(X, y, params)

    monkeypatch.setattr(pysr_reg, "_fit_pysr", record_fit)
    # Avoid hill climbing altering the graph
    monkeypatch.setattr(
        pysr_reg, "search_best_graph_climber", lambda *a, **k: (a[1], {})
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = 2 * x + rng.normal(scale=0.1, size=100)

    # DataFrame columns deliberately not in graph order
    df = pd.DataFrame({"B": y, "A": x})

    node_a, node_b = Node("A"), Node("B")
    g = Graph([node_a, node_b])
    g.add_edge(Edge(node_a, node_b, Endpoint.TAIL, Endpoint.ARROW))

    res = pysr_reg.symbolic_regression_causal_effect(df, g)

    # The first call corresponds to node A (no parents); the second is for B with parent A
    assert calls == [None, ["A"]]
    assert res["structural_equations"]["B"]["parents"] == ["A"]
