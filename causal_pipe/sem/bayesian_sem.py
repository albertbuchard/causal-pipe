from causal_pipe.utilities.model_comparison_utilities import (
    BETTER_MODEL_1,
    BETTER_MODEL_2,
    NO_BETTER_MODEL,
)


def fit_bayesian_sem_blavaan(
    data,
    model_1_string,
    var_names=None,
    model_2_string=None,
    burnin=1000,
    sample=5000,
    adapt=1000,
    n_chains=2,
):
    """
    Fits a Bayesian Structural Equation Model (SEM) using the specified model_string and returns the model summary, parameter estimates, R² values, factor scores, and per-sample log-likelihoods.

    If compare_model_string is provided, it also fits the second model and computes the Bayes Factor comparing the two models.

    Parameters:
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The dataset including all variables needed for the SEM.
    model_string : str
        The model specification string for SEM in lavaan syntax.
    var_names : dict, optional
        A dictionary mapping current factor names to meaningful names.
        Example: {'Academic': 'Academic_Ability', 'Arts': 'Artistic_Skills'}
    compare_model_string : str, optional
        The model specification string for the second SEM model to compare using Bayes Factor.
    burnin : int, optional
        Number of burn-in iterations for the MCMC sampler. Default is 1000.
    sample : int, optional
        Number of sampling iterations for the MCMC sampler. Default is 5000.
    adapt : int, optional
        Number of adaptation iterations for the MCMC sampler. Default is 1000.
    n_chains : int, optional
        Number of MCMC chains to run. Default is 2.

    Returns:
    -------
    dict
        A dictionary containing:
            - 'model_string': The SEM model specification string.
            - 'fit_summary': str, the SEM fit summary.
            - 'fit_measures': pandas.DataFrame, selected fit indices.
            - 'measurement_model': pandas.DataFrame, parameter estimates for the measurement model.
            - 'structural_model': pandas.DataFrame, parameter estimates for the structural model.
            - 'r2': pandas.DataFrame, R² values for endogenous variables.
            - 'factor_scores': pandas.DataFrame, factor scores for each participant.
            - 'log_likelihoods': pandas.Series, per-sample log-likelihoods (approximated).
            - 'npar': int, number of parameters estimated in the model.
            - 'n_samples': int, number of observations in the data.
            - 'bayes_factor': float, Bayes Factor comparing the two models (if compare_model_string is provided).
    """
    # TODO : too many errors for now
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages
    import pandas as pd
    import numpy as np
    import sys

    # Activate automatic pandas conversion
    pandas2ri.activate()

    # Import necessary R packages
    utils = rpackages.importr("utils")
    base = rpackages.importr("base")

    # Ensure the blavaan package is installed
    if not rpackages.isinstalled("blavaan"):
        utils.install_packages("blavaan")
    blavaan = rpackages.importr("blavaan")

    # Ensure that other necessary packages are installed
    required_packages = [
        "dplyr",
        "tidyr",
        "knitr",
        "mvnormalTest",
        "loo",  # For WAIC and LOOIC
        "coda",  # For MCMC diagnostics
        "bridgesampling",  # For Bayes Factor computation
    ]
    for pkg in required_packages:
        if not rpackages.isinstalled(pkg):
            utils.install_packages(pkg)

    # Import additional packages
    dplyr = rpackages.importr("dplyr")
    tidyr = rpackages.importr("tidyr")
    knitr = rpackages.importr("knitr")
    mvnormalTest = rpackages.importr("mvnormalTest")
    loo = rpackages.importr("loo")
    coda = rpackages.importr("coda")
    bridgesampling = rpackages.importr("bridgesampling")

    # Convert the pandas DataFrame or numpy array to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Assign the data and model string to R variables
    ro.globalenv["data"] = r_data
    ro.globalenv["model_string"] = model_1_string
    ro.globalenv["burnin"] = burnin
    ro.globalenv["sample"] = sample
    ro.globalenv["adapt"] = adapt
    ro.globalenv["n_chains"] = n_chains

    # Load the blavaan package in R
    ro.r("library(blavaan)")

    # Fit the Bayesian SEM model (Model 1)
    try:
        ro.r(
            """
            fit.mod <- bsem(
                model = model_string,
                data = data,
                burnin = burnin,
                sample = sample,
                adapt = adapt,
                n.chains = n_chains,
                seed = 1234,
                save.lvs = TRUE  # Save latent variable scores
            )
            """
        )
    except Exception as e:
        print("Error in fitting the Bayesian SEM model:", file=sys.stderr)
        print(e, file=sys.stderr)
        return None

    # TODO : too many errors for now
    try:
        ro.r(
            """
            summary_fit <- summary(fit.mod, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)
            """
        )
    except Exception as e:
        print(
            "Error in extracting fit summary from the Bayesian SEM model:",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return None

    # If compare_model_string is provided, fit the second model (Model 2)
    comparison_results = None
    is_better_model = None
    if model_2_string is not None:
        ro.globalenv["compare_model_string"] = model_2_string
        try:
            ro.r(
                """
                fit.mod2 <- bsem(
                    model = compare_model_string,
                    data = data,
                    cp = 'srs',  # Using default conjugate priors
                    burnin = burnin,
                    sample = sample,
                    adapt = adapt,
                    n.chains = n_chains,
                    seed = 5678,
                    save.lvs = TRUE
                )
                """
            )
        except Exception as e:
            print(
                "Error in fitting the comparison Bayesian SEM model:", file=sys.stderr
            )
            print(e, file=sys.stderr)
            return None

        # Compute the Bayes Factor using bridgesampling
        try:
            ro.r(
                """
                library(bridgesampling)
                bridge1 <- bridge_sampler(fit.mod, silent = TRUE)
                bridge2 <- bridge_sampler(fit.mod2, silent = TRUE)
                bf_result <- bf(bridge1, bridge2)
                bayes_factor <- bf_result$bf  # Bayes Factor
                """
            )
            bayes_factor = ro.r("bayes_factor")[0]
            if bayes_factor > 1:
                better_model = "Model 1"
                is_better_model = BETTER_MODEL_1
            elif bayes_factor < 1:
                better_model = "Model 2"
                is_better_model = BETTER_MODEL_2
            else:
                better_model = "None (no significant difference)"
                is_better_model = NO_BETTER_MODEL
            comparison_results = {
                "bayes_factor": bayes_factor,
                "better_model": better_model,
            }
            print(f"\nBayes Factor comparing Model 1 to Model 2: {bayes_factor}")
        except Exception as e:
            print("Error in computing Bayes Factor:", file=sys.stderr)
            print(e, file=sys.stderr)
            bayes_factor = None
    else:
        bayes_factor = None

    # Retrieve and print the SEM summary
    print("\nBayesian SEM Model Fit Summary:")
    summary_fit_str = None
    try:
        summary_fit = ro.r("capture.output(summary_fit)")
        summary_fit_str = "\n".join(list(summary_fit))
        print(summary_fit_str)
    except Exception as e:
        print("Error in retrieving the Bayesian SEM summary:", file=sys.stderr)
        print(e, file=sys.stderr)

    # Retrieve fit measures (DIC, WAIC, LOOIC)
    try:
        measures = [
            "dic",
            "p_dic",
            "waic",
            "p_waic",
            "looic",
            "p_loo",
            "npar",
        ]
        measures_str = [f"'{m}'" for m in measures]
        fit_measures = ro.r(
            f"blavInspect(fit.mod, what = c({', '.join(measures_str)}))"
        )
        fit_measures_df = pd.DataFrame(
            {"metric": measures, "value": fit_measures}
        ).set_index("metric")
        print("\nSelected Fit Measures:")
        print(fit_measures_df)
    except Exception as e:
        print("Error in retrieving fit measures:", file=sys.stderr)
        print(e, file=sys.stderr)
        fit_measures_df = None

    # The rest of the code remains the same as before...

    # Retrieve parameter estimates for the measurement model (op == "=~")
    has_latent_factors = model_1_string.count("=~") > 0
    if not has_latent_factors:
        print("No measurement model found in the model. Skipping measurement model.")
        measurement_model_df = None
    else:
        try:
            ro.r(
                """
                library(dplyr)
                library(tidyr)
                library(knitr)
                library(mvnormalTest)
                measurement_model <- parameterEstimates(fit.mod, standardized = TRUE) %>%
                    filter(op == "=~") %>%
                    select(LV = lhs, Item = rhs, Mean = post.mean, SD = post.sd, `Std.Estimate` = std.all)
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
                measurement_model_df["Item"] = measurement_model_df["Item"].map(
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

    # Retrieve parameter estimates for the structural model (op == "~")
    has_regression = model_1_string.count("~") > 0
    if not has_regression:
        print("No regression paths found in the model. Skipping structural model.")
        structural_model_df = None
    else:
        try:
            ro.r(
                """
                structural_model <- parameterEstimates(fit.mod, standardized = TRUE) %>%
                    filter(op == "~") %>%
                    select(LV = lhs, Predictor = rhs, Mean = post.mean, SD = post.sd, `Std.Estimate` = std.all)
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

    # Retrieve R2 values
    try:
        ro.r(
            """
            r2_df <- parameterEstimates(fit.mod, rsquare = TRUE) %>%
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
    if not has_latent_factors:
        factor_scores_df = None
    else:
        try:
            ro.r(
                """
                factor_scores <- lavPredict(fit.mod, method = "EBM")
                factor_scores_df <- as.data.frame(factor_scores) 
                """
            )
            factor_scores = ro.r("factor_scores_df")
            with localconverter(ro.default_converter + pandas2ri.converter):
                factor_scores_df = pandas2ri.rpy2py(factor_scores)

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

    # Extract per-sample log-likelihoods (approximated)
    log_likelihoods = None
    try:
        # Compute posterior predictive densities for each observation
        ro.r(
            """
            library(loo)
            log_lik_matrix <- blavInspect(fit.mod, 'loglik')
            log_lik <- log_colMeans_exp(log_lik_matrix)
            """
        )
        log_lik = ro.r("log_lik")
        # Convert to pandas Series
        log_likelihoods = pd.Series(log_lik, index=data.index)
    except Exception as e:
        print(
            "Error in computing per-sample log-likelihoods:",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        log_likelihoods = None

    # Get number of parameters
    npar = None
    try:
        npar = int(ro.r("blavInspect(fit.mod, 'npar')")[0])
    except Exception as e:
        print("Error in retrieving number of parameters:", file=sys.stderr)
        print(e, file=sys.stderr)

    # Compile all outputs into a dictionary
    results = {
        "model_1_string": model_1_string,
        "fit_summary": summary_fit_str if "summary_fit_str" in locals() else None,
        "fit_measures": fit_measures_df,
        "measurement_model": measurement_model_df,
        "structural_model": structural_model_df,
        "r2": r2_df,
        "factor_scores": factor_scores_df,
        "log_likelihoods": log_likelihoods,
        "npar": npar,
        "n_samples": data.shape[0],
        "bayes_factor": bayes_factor,  # Include Bayes Factor in the results
        "model_2_string": model_2_string,
        "comparison_results": comparison_results,
        "is_better_model": is_better_model,
    }

    return results
