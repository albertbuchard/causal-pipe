import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sympy import lambdify, sympify
from scipy.stats import ks_2samp
from causallearn.graph.GeneralGraph import GeneralGraph

from .utilities.utilities import dump_json_to


class CyclicSCMSimulator:
    """Simulate nonlinear cyclic structural causal models and compute fit diagnostics."""

    def __init__(
        self,
        structural_equations: Dict[str, Any],
        undirected_graph: GeneralGraph,
        df_columns: List[str],
        seed: int = 0,
    ) -> None:
        self.structural_equations = structural_equations
        self.undirected_graph = undirected_graph
        self.columns = list(df_columns)
        self.seed = seed
        self.fi_map, self.parents_of = self._parse_structural_equations(
            structural_equations, self.columns
        )
        self.components = self._build_components(undirected_graph, self.columns)

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
            func = lambdify(parents, expr, modules="numpy")
            if parents:
                fi_map[var] = lambda vals, f=func, parents=parents: float(
                    f(*[vals[p] for p in parents])
                )
            else:
                fi_map[var] = lambda vals, f=func: float(f())
            parents_of[var] = parents
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
        self, df: pd.DataFrame, out_dir: str
    ) -> Tuple[Dict[str, List[float]], Dict[Tuple[str, ...], np.ndarray], Dict[Tuple[str, ...], np.ndarray]]:
        residuals: Dict[str, List[float]] = {v: [] for v in self.columns}
        for _, row in df.iterrows():
            for v in self.columns:
                pa_vals = {p: row[p] for p in self.parents_of[v]}
                yhat = self.fi_map[v](pa_vals)
                residuals[v].append(row[v] - yhat)
        centered = {
            v: np.asarray(vals, dtype=float) - np.mean(vals)
            for v, vals in residuals.items()
        }
        Omega: Dict[Tuple[str, ...], np.ndarray] = {}
        resid_rows: Dict[Tuple[str, ...], np.ndarray] = {}
        for comp in self.components:
            Rc = np.column_stack([centered[v] for v in comp])
            Omega[tuple(comp)] = np.cov(Rc, rowvar=False)
            resid_rows[tuple(comp)] = Rc
        dump_json_to(
            {"covariances": {",".join(k): v.tolist() for k, v in Omega.items()}},
            os.path.join(out_dir, "pysr_cyclic_noise_covariances.json"),
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
                target = fi_map[v](pa_vals) + eps_draw[v]
                xn = (1 - alpha) * x[v] + alpha * target
                max_delta = max(max_delta, abs(xn - x[v]))
                x[v] = xn
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
                if noise_kind == "bootstrap":
                    idx = rng.integers(len(resid_rows[key]))
                    eps_vec = resid_rows[key][idx]
                else:
                    eps_vec = rng.multivariate_normal(
                        np.zeros(len(comp)), Omega[key]
                    )
                eps_draw = {v: eps_vec[j] for j, v in enumerate(comp)}
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
        pd.DataFrame(sim_data, columns=self.columns).to_parquet(
            os.path.join(out_dir, "simulated_dataset.parquet"), index=False
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
        real = df.values
        sim = sim_data
        conditional_r2: Dict[str, float] = {}
        conditional_rmse: Dict[str, float] = {}
        for v in self.columns:
            r = np.asarray(residuals[v])
            conditional_r2[v] = float(
                1
                - np.sum(r ** 2)
                / np.sum((df[v].values - df[v].values.mean()) ** 2)
            )
            conditional_rmse[v] = float(np.sqrt(np.mean(r ** 2)))
        mean_l2_diff = float(np.linalg.norm(sim.mean(axis=0) - real.mean(axis=0)))
        cov_frobenius_diff = float(
            np.linalg.norm(
                np.cov(sim, rowvar=False) - np.cov(real, rowvar=False),
                ord="fro",
            )
        )
        corr_frobenius_diff = float(
            np.linalg.norm(
                np.corrcoef(sim, rowvar=False)
                - np.corrcoef(real, rowvar=False),
                ord="fro",
            )
        )
        ks_pvalues = {
            v: float(ks_2samp(real[:, i], sim[:, i]).pvalue)
            for i, v in enumerate(self.columns)
        }
        q_low90 = np.quantile(sim, 0.05, axis=0)
        q_high90 = np.quantile(sim, 0.95, axis=0)
        q_low95 = np.quantile(sim, 0.025, axis=0)
        q_high95 = np.quantile(sim, 0.975, axis=0)
        coverage_90 = {}
        coverage_95 = {}
        for i, v in enumerate(self.columns):
            real_col = real[:, i]
            coverage_90[v] = float(
                np.mean((real_col >= q_low90[i]) & (real_col <= q_high90[i]))
            )
            coverage_95[v] = float(
                np.mean((real_col >= q_low95[i]) & (real_col <= q_high95[i]))
            )
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
