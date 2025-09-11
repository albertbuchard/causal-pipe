from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from sympy import Pow, log as s_log, exp as s_exp, sqrt as s_sqrt, sin as s_sin, cos as s_cos

from causal_pipe.partial_correlations.partial_correlations import get_parents_or_undirected

_EPS = 1e-12
_MAX_EXP = 50.0           # exp(±50) ~ 3.9e21; plenty before overflow
_CLIP_OUT = 1e12          # final output clamp

def _safe_log(x):
    return np.log(np.clip(x, _EPS, np.inf))

def _safe_exp(x):
    return np.exp(np.clip(x, -_MAX_EXP, _MAX_EXP))

def _safe_sqrt(x):
    return np.sqrt(np.clip(x, 0.0, np.inf))

def _safe_div(a, b):
    b2 = np.where(np.isfinite(b) & (np.abs(b) > _EPS), b, np.sign(b) * _EPS + (b==0)*_EPS)
    return a / b2

def _safe_pow(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # negative base with non-integer exponent → NaN (avoid complex)
    int_b = np.isclose(b, np.round(b))
    neg_invalid = (a < 0) & (~int_b)
    # compute via exp(b*log(|a|))
    z = b * np.log(np.clip(np.abs(a), _EPS, np.inf))
    z = np.clip(z, -_MAX_EXP, _MAX_EXP)
    out = np.exp(z)
    # restore sign for odd integer exponents with negative base
    neg_mask = (a < 0) & int_b & (np.mod(np.round(b), 2) != 0)
    out = np.where(neg_mask, -out, out)
    out = np.where(neg_invalid, np.nan, out)
    return out

# mapping for sympy.lambdify
_SAFE_MODULE = {
    "log": _safe_log,
    "exp": _safe_exp,
    "sqrt": _safe_sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "pow": _safe_pow,
    "divide": _safe_div,
}

def _safe_call(f, args):
    with np.errstate(all="ignore"):
        try:
            y = f(*args) if args else f()
        except Exception:
            return np.nan
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        return np.nan
    return float(np.clip(y, -_CLIP_OUT, _CLIP_OUT))


def _make_fi(f, parents_tuple):
    # parents_tuple is immutable; avoids late-binding surprises
    def fi(vals):
        return _safe_call(f, [vals[p] for p in parents_tuple])
    return fi

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


