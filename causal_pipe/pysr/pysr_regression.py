from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.partial_correlations.partial_correlations import get_parents_or_undirected
from causal_pipe.pysr.pysr_utilities import PySRFitterOutput
from causal_pipe.utilities.graph_utilities import copy_graph


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

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.shape[1] == 0:
        const = float(np.mean(y))
        best = {"sympy_format": str(const), "complexity": 1}
        r2 = 0.0
        return best, r2

    # skip penalty if no real predictors
    apply_penalty = penalize_absent_features and (X.shape[1] > 0)

    if apply_penalty:
        coeff = f"{penalty_coeff:.6g}" if isinstance(penalty_coeff, (int, float)) else str(penalty_coeff)

        penalty_loss_julia = lambda total_vars, coeff: fr"""
 function feature_absent_penalty(ex, dataset::Dataset{{T,L}}, options) where {{T,L}}
    # Base MSE
    pred, ok = eval_tree_array(ex, dataset.X, options)
    if !ok
        return L(Inf)
    end
    base = sum(i -> (pred[i] - dataset.y[i])^2, eachindex(pred)) / dataset.n

    # Count distinct variables
    total_vars = {total_vars}
    used = sizehint!(Set{{Int}}(), total_vars)
    foreach(ex.tree) do node  # faster version of 'for node in ex'\
        if node.degree == 0 && !node.constant
            push!(used, node.feature)
        end
    end

    miss = max(0, total_vars - length(used))
    return L(base + {coeff} * miss)
end
"""

        params = {**params, "loss_function_expression": penalty_loss_julia(X.shape[1], coeff)}

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
    pysr_params: Optional[Dict] = None
) -> PySRFitterOutput:
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
    PySRFitterOutput
        Fitting results including structural equations and the final graph.
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
        "constraints": {"pow": (-1, 1)},
    }
    params = {**default_params, **pysr_params}

    # Make a copy of the graph to avoid modifying the input
    original_graph = graph
    graph = copy_graph(graph)

    nodes = graph.nodes
    node_names = [node.get_name() for node in nodes]

    # Reindex the DataFrame to match the graph ordering.  This will raise
    # a KeyError if a graph node is missing from the data, surfacing any
    # inconsistencies early.
    df = df[node_names].copy()

    # When hill climbing orientation is disabled, treat undirected neighbors as parents
    parent_names: Dict[str, List[int]] = {}
    for target_node in nodes:
        target_name = target_node.get_name()
        predictors, pred_indices = get_parents_or_undirected(graph, graph.node_map[target_node])
        parent_names[target_name] = [n.get_name() for n in predictors]

    structural_equations: Dict[str, Dict] = {}
    for target_name, pnames in parent_names.items():
        X = df.loc[:, pnames].values if pnames else np.empty((len(df), 0))
        y = df[target_name].values
        variable_names = pnames or None
        best, r2 = _fit_pysr(X, y, params, variable_names=variable_names)
        structural_equations[target_name] = {
            "equation": best["sympy_format"] if "sympy_format" in best else str(best),
            "best": best,
            "r2": r2,
            "parents": pnames,
        }

    return PySRFitterOutput(structural_equations=structural_equations, final_graph=graph)
