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

from causal_pipe.sem.sem import (
    bootstrap_fci_edge_stability,
    bootstrap_fas_edge_stability,
    search_best_graph_climber,
)
from causallearn.search.ConstraintBased.FCI import fci
import pytest


def test_bootstrap_fci_edge_stability_returns_probabilities():
    np.random.seed(0)
    n = 100
    a = np.random.randn(n)
    b = a + np.random.randn(n) * 0.1
    c = b + np.random.randn(n) * 0.1
    data = pd.DataFrame({"A": a, "B": b, "C": c})

    probs, best_graph = bootstrap_fci_edge_stability(
        data, resamples=2, random_state=1
    )
    assert isinstance(probs, dict)
    assert all(isinstance(v, dict) for v in probs.values())
    for orient_probs in probs.values():
        assert all(0.0 <= p <= 1.0 for p in orient_probs.values())
    assert best_graph is None or isinstance(best_graph, tuple)


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


def test_hill_climb_bootstrap_returns_probabilities(monkeypatch):
    np.random.seed(0)
    n = 50
    a = np.random.randn(n)
    b = a + np.random.randn(n) * 0.1
    c = b + np.random.randn(n) * 0.1
    data = pd.DataFrame({"A": a, "B": b, "C": c})

    g, _ = fci(data.values, node_names=list(data.columns))

    def dummy_fit_sem_lavaan(*args, **kwargs):
        return {"fit_measures": {"bic": 1.0}}

    monkeypatch.setattr(
        "causal_pipe.sem.sem.fit_sem_lavaan", dummy_fit_sem_lavaan
    )

    _, best_score = search_best_graph_climber(
        data,
        g,
        max_iter=0,
        hc_bootstrap_resamples=2,
        hc_bootstrap_random_state=1,
    )

    assert "hc_edge_orientation_probabilities" in best_score
    hc_probs = best_score["hc_edge_orientation_probabilities"]
    assert isinstance(hc_probs, dict)
    for orient_probs in hc_probs.values():
        assert all(0.0 <= p <= 1.0 for p in orient_probs.values())
    assert "best_graph_with_hc_bootstrap" in best_score
    assert isinstance(best_score["best_graph_with_hc_bootstrap"], tuple)


def test_fci_bootstrap_saves_graph_with_highest_edge_probability_product(monkeypatch, tmp_path):
    data = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]})

    class MockNode:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class MockEdge:
        def __init__(self, n1, n2, e1, e2):
            self._n1 = n1
            self._n2 = n2
            self.endpoint1 = types.SimpleNamespace(name=e1)
            self.endpoint2 = types.SimpleNamespace(name=e2)

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
    g1 = MockGraph([MockEdge(A, B, "TAIL", "ARROW")])
    g2 = MockGraph(
        [MockEdge(A, B, "TAIL", "ARROW"), MockEdge(B, C, "TAIL", "ARROW")]
    )

    graphs = iter([g2, g2, g1])

    def fci_mock(*args, **kwargs):
        return next(graphs), None

    monkeypatch.setattr("causal_pipe.sem.sem.fci", fci_mock)

    captured = []

    def viz_mock(graph_obj, title, show, output_path):
        captured.append((graph_obj, title))

    monkeypatch.setattr("causal_pipe.sem.sem.visualize_graph", viz_mock)

    bootstrap_fci_edge_stability(
        data, resamples=3, random_state=0, output_dir=str(tmp_path)
    )

    assert len(captured) == 2
    first_graph, first_title = captured[0]
    second_graph, second_title = captured[1]

    assert len(first_graph.get_graph_edges()) == 1
    assert "p=1.00" in first_title
    assert len(second_graph.get_graph_edges()) == 2
    assert "p=0.67" in second_title


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
        return next(graphs), None, None

    monkeypatch.setattr("causal_pipe.sem.sem.fas", fas_mock)

    class DummyCIT:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("causal_pipe.sem.sem.CIT", DummyCIT)

    captured = []

    def viz_mock(graph_obj, title, show, output_path):
        captured.append((graph_obj, title))

    monkeypatch.setattr("causal_pipe.sem.sem.visualize_graph", viz_mock)

    bootstrap_fas_edge_stability(
        data, resamples=3, random_state=0, output_dir=str(tmp_path)
    )

    assert len(captured) == 2
    first_graph, first_title = captured[0]
    second_graph, second_title = captured[1]

    assert len(first_graph.get_graph_edges()) == 1
    assert "p=1.00" in first_title
    assert len(second_graph.get_graph_edges()) == 2
    assert "p=0.67" in second_title


def test_hc_bootstrap_saves_graph_with_highest_edge_probability_product(monkeypatch, tmp_path):
    data = pd.DataFrame({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]})

    class MockNode:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class MockEdge:
        def __init__(self, n1, n2, e1, e2):
            self._n1 = n1
            self._n2 = n2
            self.endpoint1 = types.SimpleNamespace(name=e1)
            self.endpoint2 = types.SimpleNamespace(name=e2)

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
    g1 = MockGraph([MockEdge(A, B, "TAIL", "ARROW")])
    g2 = MockGraph(
        [MockEdge(A, B, "TAIL", "ARROW"), MockEdge(B, C, "TAIL", "ARROW")]
    )

    graphs = iter([g1, g2, g2, g1])

    class DummyHillClimber:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, initial_graph, max_iter):
            return next(graphs)

    monkeypatch.setattr("causal_pipe.sem.sem.GraphHillClimber", DummyHillClimber)

    def dummy_exhaustive_results(self, general_graph, compared_to_graph=None):
        return {"fit_measures": {"bic": 1.0}}

    monkeypatch.setattr(
        "causal_pipe.sem.sem.SEMScore.exhaustive_results", dummy_exhaustive_results
    )

    captured = []

    def viz_mock(graph_obj, title, show, output_path):
        captured.append((graph_obj, title))

    monkeypatch.setattr("causal_pipe.sem.sem.visualize_graph", viz_mock)

    search_best_graph_climber(
        data,
        g1,
        max_iter=0,
        hc_bootstrap_resamples=3,
        hc_bootstrap_random_state=0,
        hc_bootstrap_output_dir=str(tmp_path),
    )

    assert len(captured) == 2
    first_graph, first_title = captured[0]
    second_graph, second_title = captured[1]

    assert len(first_graph.get_graph_edges()) == 1
    assert "p=1.00" in first_title
    assert len(second_graph.get_graph_edges()) == 2
    assert "p=0.67" in second_title
