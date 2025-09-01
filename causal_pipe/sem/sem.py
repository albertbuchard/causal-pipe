import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.sem.hill_climber import GraphHillClimber
from causal_pipe.utilities.graph_utilities import (
    dataframe_to_json_compatible_list,
    general_graph_to_sem_model,
    get_neighbors_general_graph,
)
from causal_pipe.utilities.model_comparison_utilities import (
    BETTER_MODEL_1,
    BETTER_MODEL_2,
    NO_BETTER_MODEL,
)


def format_ordered_for_sem(data: pd.DataFrame, ordered: List[str]) -> pd.DataFrame:
    """
    Format the data for SEM with ordered variables.
    """
    # Create a copy of the data
    data = data.copy()
    # Convert the ordered variables to ordered factors
    for var in ordered:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in the data.")
        data[var] = data[var].astype("category").cat.codes
    return data


def fit_sem_lavaan(
    data: pd.DataFrame,
    model_1_string: str,
    var_names: Optional[Dict[str, str]] = None,
    estimator: str = "MLM",
    model_2_string: Optional[str] = None,
    ordered: Optional[List[str]] = None,
    exogenous_vars_model_1: Optional[List[str]] = None,
    exogenous_vars_model_2: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fits a Structural Equation Model (SEM) using the specified model string and returns comprehensive results.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataset including all variables needed for the SEM.
    model_1_string : str
        The model specification string for SEM in lavaan syntax.
    var_names : Optional[Dict[str, str]]
        A dictionary mapping current factor names to meaningful names.
        Example: {'Academic': 'Academic_Ability', 'Arts': 'Artistic_Skills'}
    estimator : str, optional
        The estimator to use for fitting the SEM model. Default is "MLM", others include "MLR", "ULS", "WLSMV".
    model_2_string : Optional[str], optional
        The model specification string for the second SEM model to compare.
    ordered : Optional[List[str]], optional
        A list of variable names that are ordered (ordinal variables).

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing:
            - 'model_string': The SEM model specification string.
            - 'fit_summary': str, the SEM fit summary.
            - 'fit_measures': Dict[str, float], selected fit indices.
            - 'measurement_model': List[Dict[str, Any]], parameter estimates for the measurement model.
            - 'structural_model': List[Dict[str, Any]], parameter estimates for the structural model.
            - 'residual_covariances': List[Dict[str, Any]], residual covariances.
            - 'factor_scores': List[Dict[str, Any]], factor scores for each participant.
            - 'r2': List[Dict[str, Any]], R² values for endogenous variables.
            - 'log_likelihood': Optional[float], the total log-likelihood of the model.
            - 'log_likelihoods': Optional[pd.Series], per-sample log-likelihoods (if available).
            - 'npar': Optional[int], number of parameters estimated in the model.
            - 'n_samples': int, number of observations in the data.
            - 'comparison_results': Optional[Dict[str, Any]], model comparison results (if model_2_string is provided).
            - 'is_better_model': Optional[Any], indicator of which model is better.
            - 'model_2_string': Optional[str], the second model specification string.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages
    import sys

    # Activate automatic pandas conversion
    pandas2ri.activate()

    # Import necessary R packages
    utils = rpackages.importr("utils")
    base = rpackages.importr("base")

    # Ensure the lavaan package is installed
    if not rpackages.isinstalled("lavaan"):
        utils.install_packages("lavaan")
    lavaan = rpackages.importr("lavaan")

    # Ensure that other necessary packages are installed
    required_packages = [
        "dplyr",
        "tidyr",
        "lavaanPlot",
        "knitr",
        "mvnormalTest",
    ]
    if estimator == "ML":
        required_packages.append("nonnest2")  # Only needed if estimator is ML
    for pkg in required_packages:
        if not rpackages.isinstalled(pkg):
            utils.install_packages(pkg)

    # Import additional packages
    dplyr = rpackages.importr("dplyr")
    tidyr = rpackages.importr("tidyr")
    lavaanPlot = rpackages.importr("lavaanPlot")
    knitr = rpackages.importr("knitr")
    mvnormalTest = rpackages.importr("mvnormalTest")
    if estimator == "ML":
        nonnest2 = rpackages.importr("nonnest2")

    # Load the lavaan package in R
    ro.r("library(lavaan)")

    # Make sure no exogenous variables are included in the ordered list
    ordered_without_exogenous_model_1 = ordered
    ordered_without_exogenous_model_2 = ordered
    if ordered:
        data = format_ordered_for_sem(data, ordered=ordered)
        if exogenous_vars_model_1:
            ordered_without_exogenous_model_1 = [
                var for var in ordered if var not in exogenous_vars_model_1
            ]

        if model_2_string:
            if exogenous_vars_model_2:
                ordered_without_exogenous_model_2 = [
                    var for var in ordered if var not in exogenous_vars_model_2
                ]

    # Convert the pandas DataFrame to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Assign the data and model string to R variables
    ro.globalenv["data"] = r_data
    ro.globalenv["model_string"] = model_1_string
    ro.globalenv["estimator"] = estimator

    # Fit the SEM model with the ordered argument if provided
    try:
        if ordered_without_exogenous_model_1:
            if not isinstance(ordered, list):
                raise TypeError(
                    "The 'ordered' parameter must be a list of variable names."
                )
            ro.globalenv["ordered_vars"] = ro.StrVector(
                ordered_without_exogenous_model_1
            )
            # ro.globalenv["model_string"] = model_string_convert_residual_cov_to_factor(model_1_string)
            ro.r(
                """
                    fit.mod <- sem(model = model_string, data = data, std.lv = TRUE, 
                                   estimator = estimator, 
                                   ordered = ordered_vars)
                """
            )
        else:
            ro.r(
                """
                    fit.mod <- sem(model = model_string, data = data, std.lv = TRUE, 
                                   estimator = estimator, representation = "RAM")
                """
            )
    except Exception as e:
        print("Error in fitting the SEM model:", file=sys.stderr)
        print(e, file=sys.stderr)
        return {}

    # Extract the fit summary
    try:
        ro.r(
            """
            summary_fit <- summary(fit.mod, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)
            """
        )
        summary_fit = "\n".join(ro.r("capture.output(summary_fit)").tolist())
    except Exception as e:
        print(
            "Error in extracting fit summary from the SEM model:",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        summary_fit = ""

    # Extract non-standardized parameter estimates
    try:
        ro.r(
            """
            parameter_estimates <- parameterEstimates(fit.mod, standardized = FALSE)
            """
        )
        unstandardized_parameter_estimates = ro.r("parameter_estimates")
        unstandardized_parameter_estimates_df = pandas2ri.rpy2py(
            unstandardized_parameter_estimates
        )
        print("\nNon-Standardized Parameter Estimates:")
        print(unstandardized_parameter_estimates_df)
    except Exception as e:
        print(
            "Error in extracting non-standardized parameter estimates from the SEM model:",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        unstandardized_parameter_estimates_df = None

    # Extract standardized parameter estimates
    try:
        ro.r(
            """
            standardized_parameter_estimates <- parameterEstimates(fit.mod, standardized = TRUE)
            """
        )
        standardized_parameter_estimates = ro.r("standardized_parameter_estimates")
        standardized_parameter_estimates_df = pandas2ri.rpy2py(
            standardized_parameter_estimates
        )
        print("\nStandardized Parameter Estimates:")
        print(standardized_parameter_estimates_df)
    except Exception as e:
        print(
            "Error in extracting standardized parameter estimates from the SEM model:",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        standardized_parameter_estimates_df = None

    # Extract log-likelihood
    log_likelihood = None
    if estimator.startswith("ML"):
        try:
            log_likelihood = ro.r("as.numeric(logLik(fit.mod))")[0]
            print("\nLog-Likelihood:")
            print(log_likelihood)
        except Exception as e:
            print("Error in retrieving log-likelihood:", file=sys.stderr)
            print(e, file=sys.stderr)
    else:
        print("\nLog-Likelihood is not available for the selected estimator.")

    # Retrieve fit measures
    fit_measures = {}
    try:
        measures = [
            "bic",
            "aic",
            "df",
            "chisq.scaled",
            "df.scaled",
            "pvalue.scaled",
            "rmsea.scaled",
            "rmsea.ci.lower.scaled",
            "rmsea.ci.upper.scaled",
            "rmsea.pvalue.scaled",
            "cfi.scaled",
            "srmr",
            "npar",
        ]
        for m in measures:
            fit_measure = ro.r(f"fitMeasures(fit.mod, '{m}')")
            if fit_measure is not None and hasattr(fit_measure, "__len__") and len(fit_measure) > 0:
                fit_measures[m] = fit_measure[0]
            else:
                fit_measures[m] = fit_measure

        print("\nSelected Fit Measures:")
        print(fit_measures)
    except Exception as e:
        print("Error in retrieving fit measures:", file=sys.stderr)
        print(e, file=sys.stderr)

    # Retrieve parameter estimates for the measurement model (op == "=~")
    has_latent_factors = "=~" in model_1_string
    measurement_model_df = None
    if has_latent_factors:
        try:
            ro.r(
                """
                library(dplyr)
                library(tidyr)
                library(knitr)
                library(mvnormalTest)
                measurement_model <- standardizedSolution(fit.mod, type = "std.all", se = TRUE, zstat = TRUE, pvalue = TRUE, ci = TRUE) %>%
                    filter(op == "=~") %>%
                    select(LV = lhs, Item = rhs, Coefficient = est.std, ci.lower, ci.upper, SE = se, Z = z, `p-value` = pvalue)
                """
            )
            measurement_model = ro.r("measurement_model")
            measurement_model_df = pandas2ri.rpy2py(measurement_model)
            # Rename factor scores if var_names is provided
            if var_names is not None:
                if not isinstance(var_names, dict):
                    raise TypeError(
                        f"var_names must be a dictionary mapping current factor names to new names, got {type(var_names)}."
                    )
                measurement_model_df["LV"] = measurement_model_df["LV"].map(
                    lambda x: var_names.get(x, x)
                )

            print("\nMeasurement Model Parameter Estimates:")
            print(measurement_model_df)
        except Exception as e:
            print(
                "Error in retrieving measurement model parameter estimates:",
                file=sys.stderr,
            )
            print(e, file=sys.stderr)
            measurement_model_df = None
    else:
        print("No measurement model found in the model. Skipping measurement model.")

    # Retrieve parameter estimates for the structural model (op == "~")
    has_regression = "~" in model_1_string
    structural_model_df = None
    if has_regression:
        try:
            ro.r(
                """
                structural_model <- standardizedSolution(fit.mod, type = "std.all", se = TRUE, zstat = TRUE, pvalue = TRUE, ci = TRUE) %>%
                    filter(op == "~") %>%
                    select(LV = lhs, Predictor = rhs, Coefficient = est.std, ci.lower, ci.upper, SE = se, Z = z, `p-value` = pvalue)
                """
            )
            structural_model = ro.r("structural_model")
            with localconverter(ro.default_converter + pandas2ri.converter):
                structural_model_df = ro.conversion.rpy2py(structural_model)

            # Rename predictors if var_names is provided
            if var_names is not None:
                structural_model_df["LV"] = structural_model_df["LV"].map(
                    lambda x: var_names.get(x, x)
                )
                structural_model_df["Predictor"] = structural_model_df["Predictor"].map(
                    lambda x: var_names.get(x, x)
                )

            print("\nStructural Model Parameter Estimates:")
            print(structural_model_df)
        except Exception as e:
            print(
                "Error in retrieving structural model parameter estimates:",
                file=sys.stderr,
            )
            print(e, file=sys.stderr)
            structural_model_df = None
    else:
        print("No regression paths found in the model. Skipping structural model.")

    # Retrieve parameter estimates for the residual covariances (op == "~~")
    residual_covariances_df = None
    try:
        ro.r(
            """
            residual_covariances <- standardizedSolution(fit.mod, type = "std.all", 
                                                         se = TRUE, zstat = TRUE, 
                                                         pvalue = TRUE, ci = TRUE) %>%
                filter(op == "~~" & lhs != rhs) %>%
                select(Variable1 = lhs, Variable2 = rhs, Coefficient = est.std, 
                       ci.lower, ci.upper, SE = se, Z = z, `p-value` = pvalue)
            """
        )
        residual_covariances = ro.r("residual_covariances")
        with localconverter(ro.default_converter + pandas2ri.converter):
            residual_covariances_df = ro.conversion.rpy2py(residual_covariances)

        # Rename variables if var_names is provided
        if var_names is not None:
            residual_covariances_df["Variable1"] = residual_covariances_df[
                "Variable1"
            ].map(lambda x: var_names.get(x, x))
            residual_covariances_df["Variable2"] = residual_covariances_df[
                "Variable2"
            ].map(lambda x: var_names.get(x, x))

        print("\nResidual Covariances:")
        print(residual_covariances_df)
    except Exception as e:
        print(
            "Error in retrieving residual covariances:",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)

    # Retrieve R² values
    r2_df = None
    try:
        ro.r(
            """
            r2_df <- parameterEstimates(fit.mod, standardized = TRUE, rsquare = TRUE) %>%
                filter(op == "r2") %>%
                select(Item = lhs, R2 = est)
            """
        )
        r2 = ro.r("r2_df")
        with localconverter(ro.default_converter + pandas2ri.converter):
            r2_df = ro.conversion.rpy2py(r2)
        print("\nR² Values:")
        print(r2_df)
    except Exception as e:
        print("Error in retrieving R² values:", file=sys.stderr)
        print(e, file=sys.stderr)
        r2_df = None

    # Retrieve factor scores using lavPredict
    factor_scores_df = None
    if has_latent_factors:
        try:
            ro.r(
                """
                factor_scores <- lavPredict(fit.mod)
                factor_scores_df <- as.data.frame(factor_scores) 
                """
            )
            factor_scores = ro.r("factor_scores_df")
            with localconverter(ro.default_converter + pandas2ri.converter):
                factor_scores_df = ro.conversion.rpy2py(factor_scores)

            # Rename the columns if var_names is provided
            if var_names is not None:
                current_factor_names = factor_scores_df.columns.tolist()
                rename_mapping = {
                    current_name: var_names.get(current_name, current_name)
                    for current_name in current_factor_names
                }
                factor_scores_df.rename(columns=rename_mapping, inplace=True)

            # Align factor scores index with the original data
            factor_scores_df.reset_index(drop=True, inplace=True)
            if isinstance(data, pd.DataFrame):
                factor_scores_df.index = data.index
            elif isinstance(data, np.ndarray):
                factor_scores_df.index = range(data.shape[0])

            print("\nFactor Scores:")
            print(factor_scores_df.head())
        except Exception as e:
            print("Error in computing factor scores:", file=sys.stderr)
            print(e, file=sys.stderr)
            factor_scores_df = None
    else:
        print("No latent factors found. Skipping factor scores.")

    # Extract per-sample log-likelihoods using llcont from nonnest2 (only if estimator is ML)
    log_likelihoods = None
    if estimator == "ML":
        try:
            # Load nonnest2 package
            ro.r("library(nonnest2)")
            # Perform Vuong test
            ro.r(
                """
                llvec <- llcont(fit.mod)
                useidx <- unlist(lavInspect(fit.mod, "case.idx"))
                """
            )
            # Retrieve llvec and useidx
            llvec = np.array(ro.r("llvec"))
            useidx = np.array(ro.r("useidx"))
            # Adjust useidx to zero-based indices
            useidx = useidx - 1  # R indices start at 1
            # Create a pandas Series with per-sample log-likelihoods
            log_likelihoods = pd.Series(llvec, index=useidx)
            # Map indices to data index
            data_index = data.index.to_list()
            log_likelihoods.index = [data_index[i] for i in useidx]

            print("\nPer-sample Log-Likelihoods:")
            print(log_likelihoods.head())
        except Exception as e:
            print(
                "Error in computing per-sample log-likelihoods using llcont:",
                file=sys.stderr,
            )
            print(e, file=sys.stderr)
            log_likelihoods = None
    else:
        print(
            "\nPer-sample Log-Likelihoods are not available for the selected estimator."
        )

    # Model comparison if model_2_string is provided
    comparison_results = None
    is_better_model = None
    if model_2_string is not None:
        print("\nComparing models...")
        # Fit the second model
        ro.globalenv["model_2_string"] = model_2_string
        try:
            if ordered_without_exogenous_model_2 is not None:
                if not isinstance(ordered, list):
                    raise TypeError(
                        "The 'ordered' parameter must be a list of variable names."
                    )
                ro.globalenv["ordered_vars_model_2"] = ro.StrVector(
                    ordered_without_exogenous_model_2
                )
                ro.r(
                    """
                    fit.mod2 <- sem(model = model_2_string, data = data, std.lv = TRUE, 
                                   estimator = estimator, 
                                   ordered = ordered_vars_model_2)
                    """
                )
            else:
                ro.r(
                    """
                    fit.mod2 <- sem(model = model_2_string, data = data, std.lv = TRUE, 
                                   estimator = estimator, representation = "RAM")
                    """
                )
        except Exception as e:
            print("Error in fitting the comparison SEM model:", file=sys.stderr)
            print(e, file=sys.stderr)
            return {}

        if estimator == "ML":
            # Use nonnest2 for model comparison
            try:
                ro.r(
                    """
                    library(nonnest2)
                    # Perform Vuong test
                    vuong_test <- vuongtest(fit.mod, fit.mod2)
                    """
                )
                # Extract the elements from the vuong_test object
                omega = ro.r("vuong_test$omega")[0]
                p_omega = ro.r("vuong_test$p_omega")[0]
                nested = bool(ro.r("vuong_test$nested")[0])
                LRTstat = ro.r("vuong_test$LRTstat")[0]
                p_LRT = ro.r("vuong_test$p_LRT")
                # Convert p_LRT to a list
                with localconverter(ro.default_converter + pandas2ri.converter):
                    p_LRT = list(p_LRT)
                # Initialize better_model
                better_model = "None (no significant difference)"
                # Determine which model is better based on nested flag
                if nested:
                    # Robust likelihood ratio test
                    p_LRT_H1 = p_LRT[0]  # Only one p-value for nested models
                    if p_LRT_H1 < 0.05:
                        better_model = "Model 1"
                    else:
                        better_model = "None (no significant difference)"
                else:
                    # Non-nested likelihood ratio test
                    p_LRT_H1A = p_LRT[0]  # Model 1 better than Model 2
                    p_LRT_H1B = p_LRT[1]  # Model 2 better than Model 1
                    if p_LRT_H1A < 0.05 and p_LRT_H1B >= 0.05:
                        better_model = "Model 1"
                        is_better_model = BETTER_MODEL_1
                    elif p_LRT_H1B < 0.05 and p_LRT_H1A >= 0.05:
                        better_model = "Model 2"
                        is_better_model = BETTER_MODEL_2
                    else:
                        better_model = "None (no significant difference)"
                        is_better_model = NO_BETTER_MODEL
                # Create a dictionary with the results
                comparison_results = {
                    "omega_squared": omega,
                    "p_omega": p_omega,
                    "nested": nested,
                    "LRT_stat": LRTstat,
                }
                if nested:
                    comparison_results["p_LRT_H1"] = p_LRT_H1
                else:
                    comparison_results["p_LRT_H1A"] = p_LRT_H1A
                    comparison_results["p_LRT_H1B"] = p_LRT_H1B
                comparison_results["better_model"] = better_model

                # Print the results
                print("\nVuong Test Results:")
                print(f"Omega squared: {omega}, p-value: {p_omega}")
                if nested:
                    print(f"Models are nested.")
                    print(
                        f"Likelihood Ratio Test Statistic: {LRTstat}, p-value: {p_LRT_H1}"
                    )
                    print(f"Better model based on Vuong test: {better_model}")
                else:
                    print(f"Models are non-nested.")
                    print(f"Likelihood Ratio Test Statistic: {LRTstat}")
                    print(f"p-value for H1A (Model 1 better than Model 2): {p_LRT_H1A}")
                    print(f"p-value for H1B (Model 2 better than Model 1): {p_LRT_H1B}")
                    print(f"Better model based on Vuong test: {better_model}")
            except Exception as e:
                print("Error in performing Vuong test:", file=sys.stderr)
                print(e, file=sys.stderr)
                comparison_results = None
        else:
            # Use BIC for model comparison
            try:
                # Get BIC values
                bic1 = ro.r("fitMeasures(fit.mod, 'bic')")[0]
                bic2 = ro.r("fitMeasures(fit.mod2, 'bic')")[0]
                print(f"\nModel 1 BIC: {bic1}")
                print(f"Model 2 BIC: {bic2}")
                # Determine which model is better
                if bic1 < bic2 - 10:
                    better_model = "Model 1"
                    is_better_model = BETTER_MODEL_1
                elif bic2 < bic1 - 10:
                    better_model = "Model 2"
                    is_better_model = BETTER_MODEL_2
                else:
                    better_model = "Both models have the same BIC"
                    is_better_model = NO_BETTER_MODEL
                print(f"Better model based on BIC: {better_model}")
                # Add results to comparison_results
                comparison_results = {
                    "bic1": bic1,
                    "bic2": bic2,
                    "better_model": better_model,
                }
            except Exception as e:
                print("Error in comparing models using BIC:", file=sys.stderr)
                print(e, file=sys.stderr)
                comparison_results = None

    # Compile all outputs into a dictionary
    results = {
        "estimator": estimator,
        "model_1_string": model_1_string,
        "fit_summary": summary_fit,
        "fit_measures": fit_measures,
        "unstandardized_parameter_estimates": dataframe_to_json_compatible_list(
            unstandardized_parameter_estimates_df
        ),
        "standardized_parameter_estimates": dataframe_to_json_compatible_list(
            standardized_parameter_estimates_df
        ),
        "measurement_model": dataframe_to_json_compatible_list(measurement_model_df),
        "structural_model": dataframe_to_json_compatible_list(structural_model_df),
        "residual_covariances": dataframe_to_json_compatible_list(
            residual_covariances_df
        ),
        "factor_scores": dataframe_to_json_compatible_list(factor_scores_df),
        "r2": dataframe_to_json_compatible_list(r2_df),
        "log_likelihood": log_likelihood,
        "log_likelihoods": log_likelihoods,
        "npar": fit_measures.get("npar"),
        "n_samples": data.shape[0],
        "comparison_results": comparison_results,
        "is_better_model": is_better_model,
        "model_2_string": model_2_string,
    }

    return results


class SEMScore:
    def __init__(
        self,
        data: pd.DataFrame,
        var_names: Optional[Dict[str, str]] = None,
        estimator: str = "MLR",
        return_metrics: bool = False,
        ordered: Optional[List[str]] = None,
    ):
        """
        Initializes the SEMScore with the dataset and optional variable renaming.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset used for SEM fitting.
        var_names : Optional[Dict[str, str]]
            A dictionary mapping current factor names to meaningful names.
            Example: {'attitudes': 'Attitudes', 'norms': 'Norms'}
        estimator : str, optional
            The estimator to use for fitting the SEM model. Default is "MLM", others include "MLR", "ULS", "WLSMV".
            Or "bayesian" for Bayesian estimation using blavaan.
        return_metrics : bool, optional
            Whether to return additional fit metrics in the output.
        ordered : Optional[List[str]], optional
            A list of variable names that are ordered (ordinal variables).
        """
        self.data = data
        self.var_names = var_names
        if var_names is None:
            if isinstance(data, pd.DataFrame):
                self.var_names = list(data.columns)
            elif isinstance(data, np.ndarray):
                self.var_names = [f"Var{i}" for i in range(data.shape[1])]
        self.estimator = estimator
        if estimator == "bayesian":
            raise NotImplementedError("bayesian estimator is not yet implemented.")
        self.return_metrics = return_metrics
        if ordered:
            warnings.warn(
                "The 'ordered' parameter is not implemented yet. Ignoring it for now."
            )
            ordered = None
        self.ordered = ordered

    def __call__(
        self,
        general_graph: GeneralGraph,
        compared_to_graph: Optional[GeneralGraph] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the score of the given graph based on BIC from SEM fitting.

        Parameters
        ----------
        general_graph : GeneralGraph
            The graph to score.
        compared_to_graph : Optional[GeneralGraph], optional
            The graph to compare the given graph against.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the score and additional SEM fitting results.
        """
        results = self.exhaustive_results(
            general_graph, compared_to_graph=compared_to_graph
        )

        if results is None:
            # Assign a very low score if the model fitting failed
            return {
                "score": -np.inf,
            }

        fit_measures = results.get("fit_measures")
        if self.estimator == "bayesian":
            return {
                "score": fit_measures.loc["waic", "value"],
                "fit_measures": fit_measures,
                "is_better_model": results.get("is_better_model"),
                "comparison_results": results.get("comparison_results"),
                "all_results": results,
            }
        else:
            # Extract BIC from fit measures
            if fit_measures is None or "bic" not in fit_measures:
                # Attempt to compute BIC from AIC and log-likelihood if not directly available
                # Here, for simplicity, assign a very low score
                return {
                    "score": -np.inf,
                }

            bic = fit_measures.get("bic")

            return {
                # Return negative BIC as the score
                "score": -bic,
                "fit_measures": fit_measures,
                "is_better_model": results.get("is_better_model"),
                "comparison_results": results.get("comparison_results"),
                "all_results": results,
            }

    def exhaustive_results(
        self,
        general_graph: GeneralGraph,
        compared_to_graph: Optional[GeneralGraph] = None,
    ) -> Dict[str, Any]:
        """
        Fits an SEM model for the given graph and returns the results.

        Parameters
        ----------
        general_graph : GeneralGraph
            The graph structure to fit.
        compared_to_graph : Optional[GeneralGraph], optional
            The graph structure to compare the given graph against.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the SEM fitting results.
        """
        # Convert the graph to a SEM model string
        model_str, exogenous_vars = general_graph_to_sem_model(general_graph)

        compare_model_string = None
        if compared_to_graph is not None:
            compare_model_string, compare_exogenous_vars = general_graph_to_sem_model(
                compared_to_graph
            )
            exogenous_vars = list(set(exogenous_vars + compare_exogenous_vars))

        # Fit the SEM model
        if self.estimator == "bayesian":
            raise NotImplementedError("bayesian estimator is not yet implemented.")
            # results = fit_bayesian_sem_blavaan(
            #     data=self.data,
            #     model_1_string=model_str,
            #     var_names=None,
            #     model_2_string=compare_model_string,
            #     ordered=self.ordered,
            # )
        else:
            results = fit_sem_lavaan(
                data=self.data,
                model_1_string=model_str,
                var_names=None,
                estimator=self.estimator,
                model_2_string=compare_model_string,
                ordered=self.ordered,
                exogenous_vars_model_1=exogenous_vars,
            )

        return results


def nodes_names_from_data(data: pd.DataFrame) -> List[str]:
    """
    Extracts node names from the dataset.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset from which to extract node names.

    Returns
    -------
    List[str]
        A list of variable names.

    Raises
    ------
    ValueError
        If the data type is unsupported.
    """
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    elif isinstance(data, np.ndarray):
        return [f"Var{i}" for i in range(data.shape[1])]
    else:
        raise ValueError(
            "Unsupported data type. Please provide a pandas DataFrame or numpy array."
        )


def search_best_graph_climber(
    data: pd.DataFrame,
    initial_graph: GeneralGraph,
    node_names: Optional[List[str]] = None,
    max_iter: int = 1000,
    estimator: str = "MLM",
    finalize_with_resid_covariances: bool = False,
    mi_cutoff: float = 10.0,
    sepc_cutoff: float = 0.10,
    max_add: int = 5,
    delta_stop: float = 0.003,
    whitelist_pairs: Optional[pd.DataFrame] = None,
    forbid_pairs: Optional[pd.DataFrame] = None,
    same_occasion_regex: Optional[str] = None,
    *,
    respect_pag: bool = False,
    **kwargs,
) -> Tuple[GeneralGraph, Dict[str, Any]]:
    """
    Searches for the best graph structure using hill-climbing based on SEM fit.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset used for SEM fitting.
    initial_graph : GeneralGraph
        The initial graph structure to start the search.
    node_names : Optional[List[str]], optional
        A list of variable names in the dataset.
    max_iter : int, optional
        The maximum number of iterations for the hill-climbing search.
    estimator : str, optional
        The estimator to use for fitting the SEM model. Default is "MLM", others include "MLR", "ULS", "WLSMV".
    respect_pag : bool, optional
        When True, the search preserves PAG marks (no change to ↔, →, —; only resolves circle endpoints consistent with PAG semantics).


    Returns
    -------
    Tuple[GeneralGraph, Dict[str, Any]]
        - best_graph: The graph structure with the best SEM fit.
        - best_score: Dictionary containing SEM results. When
          ``finalize_with_resid_covariances`` is ``True`` this dictionary
          includes additional keys:

            - ``without_added_covariance_score``: the original score prior to
              any residual covariance augmentation.
            - ``resid_cov_aug``: details about the augmentation step, or ``None``
              if no covariances were added.
    """
    ordered = None

    if node_names is None:
        node_names = nodes_names_from_data(data)

    # Initialize SEMScore with the dataset and parameters
    sem_score = SEMScore(
        data=data, estimator=estimator, return_metrics=True, ordered=ordered
    )
    # Initialize the hill climber with the score function and neighbor generation function
    hill_climber = GraphHillClimber(
        score_function=sem_score,
        get_neighbors_func=get_neighbors_general_graph,
        node_names=node_names,
        keep_initially_oriented_edges=True,
        respect_pag=respect_pag,
    )

    # Run hill-climbing starting from the initial graph
    best_graph = hill_climber.run(initial_graph=initial_graph, max_iter=max_iter)
    best_score = sem_score.exhaustive_results(best_graph)
    baseline_score = best_score.copy()

    if finalize_with_resid_covariances:
        # Preserve the original score before any augmentation
        best_score["without_added_covariance_score"] = baseline_score

        model_str, _exog = general_graph_to_sem_model(best_graph)
        from causal_pipe.sem.resid_covariance_augmentation import (
            augment_residual_covariances_stepwise,
        )

        augmented = augment_residual_covariances_stepwise(
            data=data,
            model_string=model_str,
            estimator=estimator,
            max_add=max_add,
            mi_cutoff=mi_cutoff,
            sepc_cutoff=sepc_cutoff,
            delta_stop=delta_stop,
            whitelist_pairs=whitelist_pairs,
            forbid_pairs=forbid_pairs,
            same_occasion_regex=same_occasion_regex,
            verbose=True,
        )

        best_score["resid_cov_aug"] = augmented

    return best_graph, best_score
