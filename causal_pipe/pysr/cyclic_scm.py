from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sympy import lambdify, sympify, Pow
from scipy.stats import ks_2samp
from causallearn.graph.GeneralGraph import GeneralGraph
import sympy as sp
from causal_pipe.pysr.pysr_helpers import _safe_call, _SAFE_MODULE, _CLIP_OUT, _make_fi, SimulatorConfig, \
    graph_pseudolikelihood, mmd2_unbiased
from causal_pipe.pysr.pysr_utilities import PySRFitterType, PySRFitterOutput
from causal_pipe.utilities.utilities import dump_json_to


class CyclicSCMSimulator:
    """Simulate nonlinear cyclic structural causal models and compute fit diagnostics."""

    def __init__(
        self,
        structural_equations: Dict[str, Any],
        undirected_graph: GeneralGraph,
        df_columns: List[str],
        seed: int = 0,
        out_dir: str = ".",
    ) -> None:
        self.structural_equations = structural_equations
        self.undirected_graph = undirected_graph
        self.columns = list(df_columns)
        self.seed = seed
        self.fi_map, self.parents_of = self._parse_structural_equations(
            structural_equations, self.columns
        )
        self.components = self._build_components(undirected_graph, self.columns)
        self.out_dir = out_dir
        if out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _parse_structural_equations(
        equations: Dict[str, Any], columns: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        fi_map: Dict[str, Any] = {}
        parents_of: Dict[str, List[str]] = {}
        for var in columns:
            if var not in equations:
                raise ValueError(f"Missing structural equation for variable {var}")
        for var, info in equations.items():
            parents = info.get("parents", [])
            if not set(parents).issubset(columns):
                raise ValueError(f"Parents of {var} not in dataset columns")
            eq_str = info.get("equation") or info.get("sympy_format")
            expr = sympify(eq_str)

            # Drop unknown symbols → 1.0
            bad = {s for s in expr.free_symbols if str(s) not in columns}
            if bad:
                expr = expr.subs({s: 1.0 for s in bad})

            pow_fn = sp.Function("pow")
            expr = expr.replace(Pow, lambda b, e: pow_fn(b, e))

            # Parents are exactly the symbols present in the expr intersected with columns
            expr_syms = {str(s) for s in expr.free_symbols}
            parents = [c for c in columns if c in expr_syms]
            parents_t = tuple(parents)  # bind immutably

            # Use protected numeric module first, then numpy
            func = lambdify(parents_t, expr, modules=[_SAFE_MODULE, "numpy"])

            if parents_t:
                fi_map[var] = _make_fi(func, parents_t)
            else:
                # constant mechanism; evaluate once, guard, and close over the float
                c = _safe_call(func, [])
                c = 0.0 if not np.isfinite(c) else float(c)
                def fi_const(_vals, c=c):
                    return c
                fi_map[var] = fi_const

            parents_of[var] = list(parents_t)
        return fi_map, parents_of

    @staticmethod
    def _build_components(
        graph: GeneralGraph, columns: List[str]
    ) -> List[List[str]]:
        nodes = [n.get_name() for n in graph.nodes]
        adjacency: Dict[str, set] = {n: set() for n in nodes}
        for e in graph.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            adjacency[n1].add(n2)
            adjacency[n2].add(n1)
        visited: set = set()
        components: List[List[str]] = []
        for name in columns:
            if name in visited:
                continue
            stack = [name]
            comp: List[str] = []
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                stack.extend(
                    [v for v in adjacency.get(u, set()) if v not in visited]
                )
            components.append(comp)
        return components

    def estimate_noise(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, List[float]], Dict[Tuple[str, ...], np.ndarray], Dict[Tuple[str, ...], np.ndarray]]:
        residuals: Dict[str, List[float]] = {v: [] for v in self.columns}
        for _, row in df.iterrows():
            for v in self.columns:
                pa_vals = {p: row[p] for p in self.parents_of[v]}
                with np.errstate(all="ignore"):
                    yhat = self.fi_map[v](pa_vals)
                res = row[v] - yhat
                if not np.isfinite(res):
                    res = np.nan
                residuals[v].append(res)
        centered = {v: np.asarray(vals, float) for v, vals in residuals.items()}
        for v, arr in centered.items():
            mu = np.nanmean(arr)
            arr -= mu
            centered[v] = arr
        Omega: Dict[Tuple[str, ...], np.ndarray] = {}
        resid_rows: Dict[Tuple[str, ...], np.ndarray] = {}
        for comp in self.components:
            Rcols = []
            for v in comp:
                r = np.asarray(residuals[v], float)
                r = r - np.nanmean(r)
                Rcols.append(r)
            Rc = np.column_stack(Rcols)  # (n, k)
            # keep only finite rows
            mask = np.all(np.isfinite(Rc), axis=1)
            Rc_f = Rc[mask]
            k = len(comp)

            if Rc_f.shape[0] >= 2:
                cov = np.cov(Rc_f, rowvar=False, ddof=0)
            else:
                # fallback: diagonal with per-column nanvar or small jitter
                vdiag = np.nanvar(Rc, axis=0, ddof=0)
                vdiag = np.where(np.isfinite(vdiag) & (vdiag > 0), vdiag, 1e-6)
                cov = np.diag(vdiag)

            cov = np.atleast_2d(np.asarray(cov, float))
            if cov.shape != (k, k):
                cov = cov.reshape((k, k))
            cov += 1e-12 * np.eye(k)

            Omega[tuple(comp)] = cov
            resid_rows[tuple(comp)] = Rc_f if Rc_f.size else np.zeros((1, k), float)
        dump_json_to(
            {"covariances": {",".join(k): v.tolist() for k, v in Omega.items()}},
            os.path.join(self.out_dir, "pysr_cyclic_noise_covariances.json"),
        )
        return residuals, Omega, resid_rows

    @staticmethod
    def solve_component(
        comp: List[str],
        fi_map: Dict[str, Any],
        parents_of: Dict[str, List[str]],
        eps_draw: Dict[str, float],
        x_init: Dict[str, float],
        alpha: float,
        tol: float,
        max_iter: int,
    ) -> Tuple[Dict[str, float], bool, int]:
        x = dict(x_init)
        for it in range(max_iter):
            max_delta = 0.0
            for v in comp:
                pa_vals = {p: x[p] for p in parents_of[v]}
                target = fi_map[v](pa_vals)
                if not np.isfinite(target):
                    target = x[v]  # no update if invalid
                xn = (1 - alpha) * x[v] + alpha * target
                if not np.isfinite(xn):
                    xn = x[v]
                x[v] = float(np.clip(xn, -_CLIP_OUT, _CLIP_OUT))

            if max_delta < tol:
                return x, True, it + 1
        return x, False, max_iter

    def simulate(
        self,
        df: pd.DataFrame,
        Omega: Dict[Tuple[str, ...], np.ndarray],
        resid_rows: Dict[Tuple[str, ...], np.ndarray],
        out_dir: str,
        noise_kind: str = "gaussian",
        alpha: float = 0.3,
        tol: float = 1e-6,
        max_iter: int = 500,
        restarts: int = 2,
        standardized_init: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        n = len(df)
        p = len(self.columns)
        mu = (
            df.mean().to_dict()
            if not standardized_init
            else {v: 0.0 for v in self.columns}
        )
        prev = mu.copy()
        sim_data = np.zeros((n, p))
        total_calls = 0
        total_iters = 0
        failures = 0
        total_restarts = 0

        for i in range(n):
            row_vals = prev.copy()
            for comp in self.components:
                key = tuple(comp)
                k = len(comp)

                rows = resid_rows[key].shape[0]
                if noise_kind == "bootstrap" and rows == 0:
                    noise_kind = "gaussian"  # fallback

                if noise_kind == "bootstrap":
                    idx = rng.integers(rows)
                    eps_vec = resid_rows[key][idx, :]                   # (k,)
                else:
                    if k == 1:
                        std = np.sqrt(max(float(Omega[key][0, 0]), 1e-12))
                        eps_vec = rng.normal(0.0, std, size=1)
                    else:
                        eps_vec = rng.multivariate_normal(np.zeros(k), Omega[key], check_valid="ignore")
                eps_draw = {v: float(eps_vec[j]) for j, v in enumerate(comp)}
                x_init = {v: row_vals.get(v, mu[v]) for v in comp}
                alpha_local = alpha
                for attempt in range(restarts + 1):
                    sol, ok, iters = self.solve_component(
                        comp,
                        self.fi_map,
                        self.parents_of,
                        eps_draw,
                        x_init,
                        alpha=alpha_local,
                        tol=tol,
                        max_iter=max_iter,
                    )
                    if ok:
                        break
                    alpha_local *= 0.5
                row_vals.update(sol)
                total_calls += 1
                total_iters += iters
                total_restarts += attempt
                if not ok:
                    failures += 1
            prev = row_vals
            sim_data[i, :] = [row_vals[v] for v in self.columns]
        nonconv_rate = failures / total_calls if total_calls else 0.0
        avg_iters = total_iters / total_calls if total_calls else 0.0
        avg_restarts = total_restarts / total_calls if total_calls else 0.0
        dump_json_to(
            {
                "nonconvergence_rate": nonconv_rate,
                "avg_iters": avg_iters,
                "avg_restarts": avg_restarts,
            },
            os.path.join(out_dir, "pysr_cyclic_solver_stats.json"),
        )
        pd.DataFrame(sim_data, columns=self.columns).to_csv(
            os.path.join(out_dir, "pysr_cyclic_simulated_data.csv"), index=False
        )
        solver_stats = {
            "nonconvergence_rate": nonconv_rate,
            "avg_iters": avg_iters,
            "avg_restarts": avg_restarts,
        }
        return sim_data, solver_stats

    def compute_fit_measures(
            self,
            df: pd.DataFrame,
            sim_data: np.ndarray,
            residuals: Dict[str, List[float]],
            solver_stats: Dict[str, float],
    ) -> Dict[str, Any]:
        # Cast to float arrays
        real = df[self.columns].to_numpy(dtype=float)
        sim = np.asarray(sim_data, dtype=float)

        # --- R² / RMSE from residuals (NaN-safe) ---------------------------------
        conditional_r2: Dict[str, float] = {}
        conditional_rmse: Dict[str, float] = {}
        for v in self.columns:
            r = np.asarray(residuals[v], dtype=float)
            sse = float(np.nansum(r ** 2))
            y = df[v].to_numpy(dtype=float)
            sst = float(np.nansum((y - np.nanmean(y)) ** 2))
            conditional_r2[v] = 1.0 - sse / sst if sst > 0 else (1.0 if sse == 0.0 else 0.0)
            conditional_rmse[v] = float(np.sqrt(np.nanmean(r ** 2)))

        # --- Align rows with finite values for second-order stats ------------------
        mask = np.all(np.isfinite(sim), axis=1) & np.all(np.isfinite(real), axis=1)
        real_f = real[mask]
        sim_f = sim[mask]

        # mean difference of means
        if real_f.size == 0 or sim_f.size == 0:
            mean_l2_diff = float("nan")
        else:
            mean_l2_diff = float(np.linalg.norm(sim_f.mean(axis=0) - real_f.mean(axis=0)))

        # --- Covariance / correlation Frobenius diffs (robust) --------------------
        if real_f.shape[0] < 2:
            cov_frobenius_diff = float("nan")
            corr_frobenius_diff = float("nan")
        else:
            cov_frobenius_diff = float(
                np.linalg.norm(
                    np.cov(sim_f, rowvar=False) - np.cov(real_f, rowvar=False),
                    ord="fro",
                    )
            )

            def safe_corr(X: np.ndarray) -> np.ndarray:
                X = np.asarray(X, dtype=float)
                # standardize only columns with non-zero variance
                std = X.std(axis=0, ddof=0)
                nz = std > 0
                if not np.any(nz):
                    # all-constant → correlation is 0 matrix of size 1x1
                    return np.zeros((1, 1), dtype=float)
                Z = (X[:, nz] - X[:, nz].mean(axis=0)) / std[nz]
                # population covariance of standardized vars == correlation
                return (Z.T @ Z) / X.shape[0]

            corr_frobenius_diff = float(
                np.linalg.norm(safe_corr(sim_f) - safe_corr(real_f), ord="fro")
            )

        # --- KS and coverage on finite rows ---------------------------------------
        ks_pvalues: Dict[str, float] = {}
        coverage_90: Dict[str, float] = {}
        coverage_95: Dict[str, float] = {}
        n_fin = sim_f.shape[0]

        for i, v in enumerate(self.columns):
            if n_fin == 0:
                ks_pvalues[v] = float("nan")
                coverage_90[v] = float("nan")
                coverage_95[v] = float("nan")
                continue

            a = real_f[:, i]
            b = sim_f[:, i]
            ks_pvalues[v] = float(ks_2samp(a, b).pvalue) if len(a) and len(b) else float("nan")

            # quantiles on simulated; guard empty/NaN
            try:
                q_low90 = float(np.quantile(b, 0.05))
                q_high90 = float(np.quantile(b, 0.95))
                q_low95 = float(np.quantile(b, 0.025))
                q_high95 = float(np.quantile(b, 0.975))
            except Exception:
                q_low90 = q_high90 = q_low95 = q_high95 = np.nan

            if np.isfinite(q_low90) and np.isfinite(q_high90):
                coverage_90[v] = float(np.mean((a >= q_low90) & (a <= q_high90)))
            else:
                coverage_90[v] = float("nan")

            if np.isfinite(q_low95) and np.isfinite(q_high95):
                coverage_95[v] = float(np.mean((a >= q_low95) & (a <= q_high95)))
            else:
                coverage_95[v] = float("nan")

        # --- Package ---------------------------------------------------------------
        fit_measures = {
            "conditional_r2": conditional_r2,
            "conditional_rmse": conditional_rmse,
            "mean_l2_diff": mean_l2_diff,
            "cov_frobenius_diff": cov_frobenius_diff,
            "corr_frobenius_diff": corr_frobenius_diff,
            "ks_pvalues": ks_pvalues,
            "coverage_90": coverage_90,
            "coverage_95": coverage_95,
            "solver": solver_stats,
        }
        return fit_measures


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
        out_dir=sim_cfg.out_dir
    )
    out_dir = sim_cfg.out_dir or "."
    residuals, Omega, resid_rows = simulator.estimate_noise(df)
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
