import copy
import os
import warnings
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple, Callable

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.hill_climber.hill_climber import ScoreFunction, GraphHillClimber
from causal_pipe.pysr.pysr_helpers import SimulatorConfig
from causal_pipe.pysr.cyclic_scm import fit_simulate_and_score
from causal_pipe.pysr.pysr_regression import symbolic_regression_causal_effect
from causal_pipe.pysr.pysr_utilities import PySRFitterType
from causal_pipe.utilities.graph_utilities import get_neighbors_general_graph
from causal_pipe.utilities.utilities import nodes_names_from_data
from causal_pipe.utilities.model_comparison_utilities import NO_BETTER_MODEL, BETTER_MODEL_1, BETTER_MODEL_2


class PySREstimatorEnum(str, Enum):
    PSEUDOLIKELIHOOD = "pseudolikelihood"
    MMDSQUARED = "mmdsquared"

class PySRScore(ScoreFunction):
    def __init__(
        self,
        data: pd.DataFrame,
        var_names: Optional[Dict[str, str]] = None,
        estimator: str = PySREstimatorEnum.PSEUDOLIKELIHOOD,
        return_metrics: bool = False,
        fitter: Optional[PySRFitterType] = None,
        pysr_params: Optional[Dict[str, Any]] = None,
        out_dir: Optional[str] = None,
    ):
        """
        Initializes the PySRScore with data and scoring parameters.
        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be used for scoring.
        var_names : Optional[Dict[str, str]], optional
            A mapping of variable names to their descriptions.
        estimator : str, optional
            The estimator to use for scoring. Options are 'pseudolikelihood' or 'm
            mdsquared'. Default is 'pseudolikelihood'.
        return_metrics : bool, optional
            Whether to return additional metrics along with the score. Default is False.
        """
        super().__init__()
        self.data = data
        self.var_names = var_names
        if var_names is None:
            if isinstance(data, pd.DataFrame):
                self.var_names = list(data.columns)
            elif isinstance(data, np.ndarray):
                self.var_names = [f"Var{i}" for i in range(data.shape[1])]
                warnings.warn(
                    "[PySRScore] var_names not provided for ndarray data; using Var{i} names."
                )
        self.estimator = estimator
        if estimator not in {e.value for e in PySREstimatorEnum}:
            raise ValueError(
                f"Invalid estimator '{estimator}'. Must be one of {[e.value for e in PySREstimatorEnum]}."
            )
        self.return_metrics = return_metrics
        self.fitter = fitter or symbolic_regression_causal_effect
        self.pysr_params = pysr_params or {}
        self.out_dir = out_dir
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)


    def __call__(
        self,
        model_1: GeneralGraph,
        model_2: Optional[GeneralGraph] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the score for the given graph using PySR fitting.

        Parameters
        ----------
        model_1 : GeneralGraph
            The graph to score.
        model_2 : Optional[GeneralGraph], optional
            The graph to compare the given graph against.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the score and additional metrics.
        """
        results = self.exhaustive_results(
            model_1, model_2=model_2
        )

        if not results:
            # Assign a very low score if the model fitting failed
            warnings.warn("[PySRScore] No results returned from PySR fitting.")
            return {
                "score": -np.inf,
                "is_better_model": NO_BETTER_MODEL,
            }

        fit_measures = results.get("fit_measures")
        if fit_measures is None:
            warnings.warn("[PySRScore] No fit measures returned from PySR fitting.")
            return {
                "score": -np.inf,
                "is_better_model": NO_BETTER_MODEL,
            }

        is_better_model = results.get("is_better_model")
        comparison_results = results.get("comparison_results")
        score = -np.inf
        if self.estimator == PySREstimatorEnum.MMDSQUARED:
            mmd_squared = fit_measures.get("mmd_squared") if fit_measures else None
            if mmd_squared is None:
                warnings.warn("[PySRScore] No mmd_squared returned from PySR fitting.")
            else:
                score = -mmd_squared
        elif self.estimator == PySREstimatorEnum.PSEUDOLIKELIHOOD:
            pseudolikelihood = fit_measures.get("pseudolikelihood") if fit_measures else None
            if pseudolikelihood is None:
                warnings.warn("[PySRScore] No pseudolikelihood returned from PySR fitting.")
            else:
                score = pseudolikelihood
        else:
            raise ValueError(f"Unknown estimator '{self.estimator}'.")

        return {
            "score": score,
            "fit_measures": fit_measures,
            "is_better_model": is_better_model,
            "comparison_results": comparison_results,
            "all_results": results,
        }

    def exhaustive_results(
        self,
        model_1: GeneralGraph,
        model_2: Optional[GeneralGraph] = None,
        exogenous_residual_covariances: bool = False,  # kept for API symmetry
    ) -> Dict[str, Any]:
        """
        Fit PySR equations for model_1 (and optionally model_2), simulate, and
        compute diagnostics + the requested estimator (pseudolikelihood or mmd^2).
        """
        # Ensure DataFrame with proper column order
        if isinstance(self.data, pd.DataFrame):
            df = self.data[self.var_names].copy()
        else:
            df = pd.DataFrame(self.data, columns=self.var_names)

        sim_cfg = SimulatorConfig(
            noise_kind="gaussian",  # use 'bootstrap' if you want empirical tails
            alpha=0.3, tol=1e-6, max_iter=500, restarts=2,
            standardized_init=False, seed=0,
            out_dir=self.out_dir
        )

        # Model 1
        _, diag1, meta1 = fit_simulate_and_score(
            df=df,
            graph=model_1,
            fitter=self.fitter or symbolic_regression_causal_effect,
            pysr_params=self.pysr_params or {},
            sim_cfg=sim_cfg,
        )

        results = {
            "fit_measures": diag1,
            "structural_equations": meta1["structural_equations"],
            "solver": meta1["solver"],
            "is_better_model": NO_BETTER_MODEL,
            "comparison_results": None,
        }

        if model_2 is None:
            return results

        # Model 2 (comparison)
        _, diag2, meta2 = fit_simulate_and_score(
            df=df,
            graph=model_2,
            fitter=self.fitter or symbolic_regression_causal_effect,
            pysr_params=self.pysr_params or {},
            sim_cfg=sim_cfg,
        )

        # Decide which is better per estimator
        if self.estimator == PySREstimatorEnum.PSEUDOLIKELIHOOD:
            s1 = diag1["pseudolikelihood"]
            s2 = diag2["pseudolikelihood"]
            is_better = BETTER_MODEL_1 if s1 > s2 else BETTER_MODEL_2 if s2 > s1 else NO_BETTER_MODEL
            comp = {"model_1_pseudolikelihood": s1, "model_2_pseudolikelihood": s2}
        elif self.estimator == PySREstimatorEnum.MMDSQUARED:
            s1 = diag1["mmd_squared"]
            s2 = diag2["mmd_squared"]
            is_better = BETTER_MODEL_1 if s1 < s2 else BETTER_MODEL_2 if s2 < s1 else NO_BETTER_MODEL
            comp = {"model_1_mmd_squared": s1, "model_2_mmd_squared": s2}
        else:
            warnings.warn(f"[PySRScore] Unknown estimator {self.estimator}; returning NO_BETTER_MODEL.")
            is_better, comp = NO_BETTER_MODEL, None

        results.update({
            "is_better_model": is_better,
            "comparison_results": comp,
            "fit_measures_model_2": diag2,
            "structural_equations_model_2": meta2["structural_equations"],
            "solver_model_2": meta2["solver"],
        })
        return results



def search_best_graph_climber_pysr(
    data: pd.DataFrame,
    *,
    initial_graph: GeneralGraph,
    node_names: Optional[List[str]] = None,
    max_iter: int = 1000,
    estimator: str = PySREstimatorEnum.PSEUDOLIKELIHOOD,
    respect_pag: bool = True,
    pysr_params=None,
    out_dir: Optional[str] = None,
) -> Tuple[GeneralGraph, Dict[str, Any]]:
    """
    Searches for the best graph structure using hill-climbing based on PySR Cyclic SCM fitting.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset used for SEM fitting.
    initial_graph : GeneralGraph
        The starting graph structure for hill-climbing.
    node_names : Optional[List[str]], optional
        List of variable names corresponding to the columns in `data`.
        If `data` is a DataFrame, this can be omitted. Default is None.
    max_iter : int, optional
        Maximum number of hill-climbing iterations. Default is 1000.
    estimator : str, optional
        The estimator to use for scoring. Options are 'pseudolikelihood' or 'm
        mdsquared'. Default is 'pseudolikelihood'.
    respect_pag : bool, optional
        Whether to respect the orientations in the initial PAG. Default is True.

    Returns
    -------
    Tuple[GeneralGraph, Dict[str, Any]]
        - best_graph: The graph structure with the best SEM fit.
        - best_score: Dictionary containing the best score and additional metrics.
    """
    if node_names is None:
        node_names = nodes_names_from_data(data)

    # Initialize SEMScore with the dataset and parameters
    scorer = PySRScore(
        data=data,
        estimator=estimator,
        return_metrics=True,
        pysr_params=pysr_params,
        out_dir=out_dir
    )
    # Initialize the hill climber with the score function and neighbor generation function
    hill_climber = GraphHillClimber(
        score_function=scorer,
        get_neighbors_func=get_neighbors_general_graph,
        node_names=node_names,
        keep_initially_oriented_edges=True,
        respect_pag=respect_pag,
        name="PySR Hill Climber",
    )

    # Run hill-climbing starting from the initial graph
    initial_graph_copy = copy.deepcopy(initial_graph)
    best_graph = hill_climber.run(initial_graph=initial_graph_copy, max_iter=max_iter)
    best_score = scorer.exhaustive_results(best_graph)

    if best_graph is None:
        raise RuntimeError("Hill climbing did not produce a best graph.")

    return best_graph, best_score
