import json
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from causallearn.graph.Endpoint import Endpoint

from causallearn.graph.GeneralGraph import GeneralGraph

from .partial_correlations.partial_correlations import get_parents_or_undirected
from .sem.sem import search_best_graph_climber
from .utilities.graph_utilities import is_fully_oriented, both_circles, copy_graph, unify_edge_types_directed_undirected


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
    pysr_params: Optional[Dict] = None,
    hc_orient_undirected_edges: bool = True,
    respect_pag: bool = True,
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

    edge_tests: Dict[str, Dict] = {}
    if hc_orient_undirected_edges:
        if not respect_pag:
            raise NotImplementedError("Only Hill Climbing with PAG respect is implemented for PySR orientation.")


        for i, target_node in enumerate(nodes):
            target_name = target_node.get_name()
            t_idx = graph.node_map[target_node]
            predictors, _ = get_parents_or_undirected(graph, t_idx)
            if not predictors:
                continue

            y = df.loc[:, target_name].values

            # Evaluate each predictor for orientation suggestions
            for predictor in predictors:
                # Check for predictors that could be children in the PAG
                p_idx = graph.node_map[predictor]
                p_name = predictor.get_name()

                edge = graph.get_edge(predictor, target_node)
                if edge is None:
                    continue

                # The only case we would run the test is there is uncertainty about an edge between p - t
                # if t -> p is in the PAG then p could in fact be a child of t, and we would run the orientation test
                if is_fully_oriented(edge):
                    # No uncertainty to resolve in the PAG for p - t
                    continue

                if not both_circles(edge) and edge.get_endpoint1() != Endpoint.CIRCLE:
                    # Could be a parent or child or confounded if both circles,
                    # if both_circles(edge) p o-o t
                    # Could be p -> t, t -> p, or p <-> t, so we run the orientation test
                    # If edge.get_endpoint1() == Endpoint.CIRCLE: p o- t
                    # Could be t -> p or p <-> t so we run the orientation test
                    # But if edge.get_endpoint2() == Endpoint.CIRCLE: p -o t
                    # Could be p -> t or p <-> t, not t -> p, we pass, p is not a child
                    continue




                # if graph.is_directed_from_to(predictor, target_node):
                #     # Already oriented p -> target; no need for test
                #     continue
                # if graph.is_directed_from_to(target_node, predictor):
                #     # target -> p; skip since p is child
                #     continue
                #
                # with_idx = pred_indices
                # without_idx = [idx for idx in pred_indices if idx != p_idx]

                with_names = [n.get_name() for n in predictors]
                without_names = [n for n in with_names if n != p_name]


                X_with = df.loc[:, with_names].values
                _, r2_with = _fit_pysr(X_with, y, params)

                if without_names:
                    X_without = df.loc[:, without_names].values
                else:
                    X_without = np.empty((len(df), 0))
                _, r2_without = _fit_pysr(X_without, y, params)
                improvement_target = r2_with - r2_without

                # Symmetric test: predictor as target
                ppredictors, ppred_indices = get_parents_or_undirected(graph, p_idx)
                ppredictors_names = [n.get_name() for n in ppredictors]
                if target_name not in ppredictors_names:
                    ppredictors.append(target_node)

                yp = df.loc[:, p_name].values

                Xp_with = df.loc[:, ppredictors_names].values
                _, r2p_with = _fit_pysr(Xp_with, yp, params)

                ppred_without = [n for n in ppredictors_names if n != target_name]
                if ppred_without:
                    Xp_without = df.loc[:, ppred_without].values
                else:
                    Xp_without = np.empty((len(df), 0))
                _, r2p_without = _fit_pysr(Xp_without, yp, params)
                improvement_pred = r2p_with - r2p_without

                # Orient p -> target if target improves and not vice versa
                if improvement_target > 0 >= improvement_pred:
                    orientation = f"{p_name} -> {target_name}"
                # Orient target -> p if p improves and not target
                elif improvement_pred > 0 >= improvement_target:
                    orientation = f"{target_name} -> {p_name}"
                else:
                    # If both improve or neither, keep as undirected
                    continue

                edge_tests[f"{p_name}--{target_name}"] = {
                    "improvement_parent_to_child": improvement_target,
                    "improvement_child_to_parent": improvement_pred,
                    "suggested_orientation": orientation,
                }

        # Determine final parent sets based on orientation suggestions
        # parent_names: Dict[str, List[str]] = {}
        # for node in nodes:
        #     name = node.get_name()
        #     # include existing directed parents from graph
        #     for parent in graph.get_parents(node):
        #         parent_names[name].append(parent.get_name())

        # for info in edge_tests.values():
        #     src, dst = info["suggested_orientation"].split(" -> ")
        #     dst_names = parent_names[dst]
        #     if src not in dst_names:
        #         dst_names.append(src)

        # Update the graph with the suggested orientations
        for info in edge_tests.values():
            src, dst = info["suggested_orientation"].split(" -> ")
            src_node = graph.get_node(src)
            dst_node = graph.get_node(dst)
            if src_node is None or dst_node is None:
                raise ValueError(f"Nodes {src} or {dst} not found in graph during orientation update.")
            # Get edge and update
            edge = graph.get_edge(src_node, dst_node)
            if edge is None:
                raise ValueError(f"Edge {src} - {dst} not found in graph during orientation update.")
            graph.remove_edge(edge)
            graph.add_directed_edge(src_node, dst_node)

        # # Remove CIRCLE edges
        # graph = unify_edge_types_directed_undirected(graph)
        # parent_names: Dict[str, List[str]] = {}
        # for node in nodes:
        #     name = node.get_name()
        #     # include existing directed parents from graph
        #     for parent in graph.get_parents(node):
        #         parent_names[name].append(parent.get_name())
        #
        # structural_equations: Dict[str, Dict] = {}
        # for target_name, pnames in parent_names.items():
        #     X = df.loc[:, pnames].values if pnames else np.empty((len(df), 0))
        #     y = df[target_name].values
        #     variable_names = pnames or None
        #     best, r2 = _fit_pysr(X, y, params, variable_names=variable_names)
        #     structural_equations[target_name] = {
        #         "equation": best["sympy_format"] if "sympy_format" in best else str(best),
        #         "best": best,
        #         "r2": r2,
        #         "parents": pnames,
        #     }

        # return {"edge_tests": edge_tests, "structural_equations": structural_equations, "final_graph": graph}

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

    return {"edge_tests": edge_tests, "structural_equations": structural_equations, "final_graph": graph}
