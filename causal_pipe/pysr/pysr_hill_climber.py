import copy
import warnings
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple, Callable

import numpy as np
import pandas as pd
from causallearn.graph import GeneralGraph

from causal_pipe.hill_climber.hill_climber import ScoreFunction, GraphHillClimber
from causal_pipe.pysr.pysr_regression import symbolic_regression_causal_effect
from causal_pipe.pysr.pysr_utilities import PySRFitterType
from causal_pipe.utilities.graph_utilities import get_neighbors_general_graph
from causal_pipe.utilities.utilities import nodes_names_from_data
from causal_pipe.utilities.model_comparison_utilities import NO_BETTER_MODEL


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
        exogenous_residual_covariances: bool = False,
    ) -> Dict[str, Any]:
        """
        Fits an PySR Cyclic SCM to the data and returns the fitting results.

        Parameters
        ----------
        model_1 : GeneralGraph
            The graph structure to fit.
        model_2 : Optional[GeneralGraph], optional
            The graph structure to compare the given graph against.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the fitting results and metrics.
        """
        # Convert the graph to a SEM model string
        results = None

        return results



def search_best_graph_climber_pysr(
    data: pd.DataFrame,
    *,
    initial_graph: GeneralGraph,
    node_names: Optional[List[str]] = None,
    max_iter: int = 1000,
    estimator: str = PySREstimatorEnum.PSEUDOLIKELIHOOD,
    respect_pag: bool = True,
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
    sem_score = PySRScore(
        data=data, estimator=estimator, return_metrics=True
    )
    # Initialize the hill climber with the score function and neighbor generation function
    hill_climber = GraphHillClimber(
        score_function=sem_score,
        get_neighbors_func=get_neighbors_general_graph,
        node_names=node_names,
        keep_initially_oriented_edges=True,
        respect_pag=respect_pag,
        name="PySR Hill Climber",
    )

    # Run hill-climbing starting from the initial graph
    initial_graph_copy = copy.deepcopy(initial_graph)
    best_graph = hill_climber.run(initial_graph=initial_graph_copy, max_iter=max_iter)
    best_score = sem_score.exhaustive_results(best_graph)

    if best_graph is None:
        raise RuntimeError("Hill climbing did not produce a best graph.")

    return best_graph, best_score
