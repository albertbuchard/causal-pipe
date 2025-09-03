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

    probs = bootstrap_fci_edge_stability(data, resamples=2, random_state=1)
    assert isinstance(probs, dict)
    assert all(isinstance(v, dict) for v in probs.values())
    for orient_probs in probs.values():
        assert all(0.0 <= p <= 1.0 for p in orient_probs.values())


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
