import json
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph

from .partial_correlations.partial_correlations import get_parents_or_undirected
from .sem.sem import search_best_graph_climber


def _fit_pysr(X: np.ndarray,
              y: np.ndarray,
              params: Dict,
              variable_names: Optional[List[str]] = None,
              penalize_absent_features: bool = True,
              penalty_coeff: Union[str, float] = 1e3
              ) -> Tuple[Dict[str, Any], float]:
    """Fit a PySR symbolic regression model and return equation string and R^2."""
    try:
        from pysr import PySRRegressor, jl
        # jl.seval('import Pkg; Pkg.add("DynamicExpressions"); using DynamicExpressions')
    except ImportError as exc:
        raise ImportError("PySR is required for symbolic regression causal effect estimation") from exc

    # Ensure X has at least one column
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        # Use a constant column when no predictors are provided
        X = np.ones((len(X), 1))

    # skip penalty if no real predictors
    apply_penalty = penalize_absent_features and (X.shape[1] > 0)

    if apply_penalty:
        coeff = f"{penalty_coeff:.6g}" if isinstance(penalty_coeff, (int, float)) else str(penalty_coeff)

        penalty_code = lambda total_vars, coeff: fr"""
feature_absent_penalty(ex, dataset, options) = begin
    # base MSE
    pred, ok = eval_tree_array(ex, dataset.X, options)
    if !ok
        return Inf
    end
    base = sum((pred .- dataset.y).^2) / dataset.n

    # count distinct variable leaves
    used = Set{{Int}}()
    function walk(n)
        if n.degree == 0
            if !n.constant
                push!(used, Int(n.feature))
            end
        elseif n.degree == 1
            walk(n.l)
        else
            walk(n.l); walk(n.r)
        end
    end
    walk(ex.tree)

    missing = max(0, {total_vars} - length(used))
    return base + {coeff} * missing
end
"""
        params = {**params, "loss_function_expression": penalty_code(X.shape[1], coeff)}

    model = PySRRegressor(**params)
    model.fit(X, y, variable_names=variable_names)
    best = {}
    try:
        best = { k: v for k, v in model.get_best().to_dict().items() if k != "lambda_format" }
        best["latex"] = model.latex()
    except Exception:
        # Fallback if get_best is not available
        best["sympy_format"] = str(model.get_best()["sympy_format"])
    r2 = model.score(X, y)
    return best, float(r2)


def symbolic_regression_causal_effect(
    df: pd.DataFrame,
    graph: GeneralGraph,
    pysr_params: Dict | None = None,
    hc_orient_undirected_edges: bool = True,
) -> Dict:
    """Estimate causal mechanisms using symbolic regression via PySR.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data.
    graph : GeneralGraph
        Graph containing directed and undirected edges.
    pysr_params : dict, optional
        Parameters for :class:`pysr.PySRRegressor`.
    hc_orient_undirected_edges : bool, optional
        When ``True`` (default), attempt to orient undirected edges using
        SEM hill climbing before running PySR. When ``False``, undirected
        edges are treated as parents for both incident nodes and no
        orientation tests are performed.

    Returns
    -------
    dict
        Dictionary containing equations for each node and orientation tests
        for edges that were not oriented in ``graph``.
    """
    if pysr_params is None:
        pysr_params = {}

    default_params = {
        # Broad search space similar to PySR defaults
        "niterations": 200,
        "population_size": 200,
        "binary_operators": ["+", "-", "*", "/", "pow"],
        "unary_operators": ["exp", "log", "sin", "cos", "sqrt"],
        "maxsize": 20,
        "maxdepth": 5,
    }
    params = {**default_params, **pysr_params}

    node_names = list(df.columns)

    if hc_orient_undirected_edges:
        # Attempt to orient edges via hill climbing prior to PySR fits
        try:
            graph, _ = search_best_graph_climber(
                data=df,
                initial_graph=graph,
                node_names=node_names,
                respect_pag=True,
            )
        except Exception:
            # If hill climbing fails, continue with original graph
            pass

        edge_tests: Dict[str, Dict] = {}

        n_nodes = len(node_names)
        for i in range(n_nodes):
            target = node_names[i]
            predictors, pred_indices = get_parents_or_undirected(graph, i)
            if not pred_indices:
                continue

            y = df.iloc[:, i].values

            # Evaluate each predictor for orientation suggestions
            for p_idx in pred_indices:
                if p_idx <= i:
                    # Avoid duplicate testing of undirected edges
                    continue
                p_name = node_names[p_idx]
                if graph.is_directed_from_to(graph.nodes[p_idx], graph.nodes[i]):
                    # Already oriented p -> target; no need for test
                    continue
                if graph.is_directed_from_to(graph.nodes[i], graph.nodes[p_idx]):
                    # target -> p; skip since p is child
                    continue

                with_idx = pred_indices
                without_idx = [idx for idx in pred_indices if idx != p_idx]

                X_with = df.iloc[:, with_idx].values
                _, r2_with = _fit_pysr(X_with, y, params)

                if without_idx:
                    X_without = df.iloc[:, without_idx].values
                else:
                    X_without = np.empty((len(df), 0))
                _, r2_without = _fit_pysr(X_without, y, params)
                improvement_target = r2_with - r2_without

                # Symmetric test: predictor as target
                ppredictors, ppred_indices = get_parents_or_undirected(graph, p_idx)
                if i not in ppred_indices:
                    ppred_indices.append(i)
                yp = df.iloc[:, p_idx].values
                Xp_with = df.iloc[:, ppred_indices].values
                _, r2p_with = _fit_pysr(Xp_with, yp, params)
                ppred_without = [idx for idx in ppred_indices if idx != i]
                if ppred_without:
                    Xp_without = df.iloc[:, ppred_without].values
                else:
                    Xp_without = np.empty((len(df), 0))
                _, r2p_without = _fit_pysr(Xp_without, yp, params)
                improvement_pred = r2p_with - r2p_without

                orientation = (
                    f"{p_name} -> {target}"
                    if improvement_target >= improvement_pred
                    else f"{target} -> {p_name}"
                )
                edge_tests[f"{p_name}--{target}"] = {
                    "improvement_parent_to_child": improvement_target,
                    "improvement_child_to_parent": improvement_pred,
                    "suggested_orientation": orientation,
                }

        # Determine final parent sets based on orientation suggestions
        parent_indices: Dict[str, List[int]] = {name: [] for name in node_names}
        for i, name in enumerate(node_names):
            # include existing directed parents from graph
            for parent in graph.get_parents(graph.nodes[i]):
                parent_indices[name].append(graph.node_map[parent])

        for info in edge_tests.values():
            src, dst = info["suggested_orientation"].split(" -> ")
            dst_idxs = parent_indices[dst]
            src_idx = node_names.index(src)
            if src_idx not in dst_idxs:
                dst_idxs.append(src_idx)

        structural_equations: Dict[str, Dict] = {}
        for name, pidxs in parent_indices.items():
            X = df.iloc[:, pidxs].values if pidxs else np.empty((len(df), 0))
            y = df[name].values
            variable_names = [node_names[idx] for idx in pidxs] if pidxs else None
            best, r2 = _fit_pysr(X, y, params, variable_names=variable_names)
            structural_equations[name] = {
                "equation": best["sympy_format"] if "sympy_format" in best else str(best),
                "best": best,
                "r2": r2,
                "parents": [node_names[idx] for idx in pidxs],
            }

        return {"edge_tests": edge_tests, "structural_equations": structural_equations}

    # When hill climbing orientation is disabled, treat undirected neighbors as parents
    parent_indices: Dict[str, List[int]] = {name: [] for name in node_names}
    for i, name in enumerate(node_names):
        _, pred_indices = get_parents_or_undirected(graph, i)
        parent_indices[name] = pred_indices

    structural_equations: Dict[str, Dict] = {}
    for name, pidxs in parent_indices.items():
        X = df.iloc[:, pidxs].values if pidxs else np.empty((len(df), 0))
        y = df[name].values
        variable_names = [node_names[idx] for idx in pidxs] if pidxs else None
        best, r2 = _fit_pysr(X, y, params, variable_names=variable_names)
        structural_equations[name] = {
            "equation": best["sympy_format"] if "sympy_format" in best else str(best),
            "best": best,
            "r2": r2,
            "parents": [node_names[idx] for idx in pidxs],
        }

    return {"edge_tests": {}, "structural_equations": structural_equations}
