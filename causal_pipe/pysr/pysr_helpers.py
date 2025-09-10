from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.partial_correlations.partial_correlations import get_parents_or_undirected
from causal_pipe.pysr.cyclic_scm import CyclicSCMSimulator
from causal_pipe.utilities.utilities import dump_json_to
from causal_pipe.pysr.pysr_utilities import PySRFitterType
from causal_pipe.pysr.pysr_utilities import PySRFitterOutput


# ---------- Graph parents ----------------------------------------------------

def parent_names_from_graph(graph: GeneralGraph) -> Dict[str, List[str]]:
    """
    Use the provided parent function to collect parents or undirected neighbors.
    """
    parent_names: Dict[str, List[str]] = {}
    nodes = graph.nodes
    for target_node in nodes:
        target_name = target_node.get_name()
        predictors, _ = get_parents_or_undirected(graph, graph.node_map[target_node])
        parent_names[target_name] = [n.get_name() for n in predictors]
    return parent_names


# ---------- Pseudo log-likelihood -------------------------------------------

def gaussian_pll_from_residuals(resid: np.ndarray) -> float:
    """
    Node-wise Gaussian pseudo log-likelihood for residuals e ~ N(0, sigma^2).
    """
    var = max(float(resid.var(ddof=0)), 1e-12)
    n = resid.shape[0]
    return -0.5 * n * (math.log(2.0 * math.pi * var) + 1.0)


def graph_pseudolikelihood(residuals: Dict[str, List[float]]) -> float:
    """
    Sum Gaussian PLL across nodes.
    """
    pll = 0.0
    for v, r in residuals.items():
        pll += gaussian_pll_from_residuals(np.asarray(r, dtype=float))
    return float(pll)


# ---------- MMD^2 (unbiased) -------------------------------------------------

def _median_bandwidth(Z: np.ndarray, max_pairs: int = 20000) -> float:
    n = Z.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.default_rng(0)
    if n * (n - 1) // 2 > max_pairs:
        idx = rng.choice(n, size=min(2 * int(max_pairs ** 0.5), n), replace=False)
        Zs = Z[idx]
    else:
        Zs = Z
    d2 = []
    for i in range(Zs.shape[0] - 1):
        diff = Zs[i + 1:] - Zs[i]
        if diff.size:
            d2.extend(np.sum(diff * diff, axis=1))
    med = np.median(d2) if d2 else 1.0
    return max(float(med), 1e-6)


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Unbiased MMD^2 with RBF kernel. Lower is better.
    """
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0
    if gamma is None:
        bw = _median_bandwidth(np.vstack([X, Y]))
        gamma = 1.0 / (2.0 * bw)

    def k_rbf(A: np.ndarray) -> np.ndarray:
        sq = np.sum(A * A, axis=1, keepdims=True)
        D = sq + sq.T - 2.0 * (A @ A.T)
        return np.exp(-gamma * D)

    Kxx = k_rbf(X)
    Kyy = k_rbf(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))

    sqX = np.sum(X * X, axis=1)[:, None]
    sqY = np.sum(Y * Y, axis=1)[None, :]
    Kxy = np.exp(-gamma * (sqX + sqY - 2.0 * (X @ Y.T)))
    term_xy = 2.0 * Kxy.mean()
    return float(term_xx + term_yy - term_xy)


# ---------- Fit + simulate wrapper ------------------------------------------

@dataclass
class SimulatorConfig:
    noise_kind: str = "gaussian"        # 'gaussian' | 'bootstrap'
    alpha: float = 0.3
    tol: float = 1e-6
    max_iter: int = 500
    restarts: int = 2
    standardized_init: bool = False
    seed: int = 0
    out_dir: Optional[str] = None


def fit_simulate_and_score(
    df: pd.DataFrame,
    graph: GeneralGraph,
    fitter: PySRFitterType,
    pysr_params: Dict[str, Any],
    sim_cfg: SimulatorConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    1) Fit PySR equations conditioned on `graph`
    2) Simulate from the cyclic SCM
    3) Compute diagnostics + scores: pseudolikelihood and MMD^2
    """
    # Fit equations
    fit_out: PySRFitterOutput = fitter(df, graph, pysr_params)
    structural_equations = fit_out.structural_equations

    # Simulate
    simulator = CyclicSCMSimulator(
        structural_equations=structural_equations,
        undirected_graph=graph,                 # components from the same graph
        df_columns=list(df.columns),
        seed=sim_cfg.seed,
    )
    out_dir = sim_cfg.out_dir or "."
    residuals, Omega, resid_rows = simulator.estimate_noise(df, out_dir)
    sim_data, solver_stats = simulator.simulate(
        df,
        Omega=Omega,
        resid_rows=resid_rows,
        out_dir=out_dir,
        noise_kind=sim_cfg.noise_kind,
        alpha=sim_cfg.alpha,
        tol=sim_cfg.tol,
        max_iter=sim_cfg.max_iter,
        restarts=sim_cfg.restarts,
        standardized_init=sim_cfg.standardized_init,
    )
    diagnostics = simulator.compute_fit_measures(df, sim_data, residuals, solver_stats)

    # Scores
    pll = graph_pseudolikelihood(residuals)
    X = df.values.astype(float, copy=False)
    mmd2 = mmd2_unbiased(X, sim_data.astype(float, copy=False))

    diagnostics["pseudolikelihood"] = float(pll)
    diagnostics["mmd_squared"] = float(mmd2)

    meta = {
        "solver": solver_stats,
        "structural_equations": structural_equations,
    }
    return structural_equations, diagnostics, meta
