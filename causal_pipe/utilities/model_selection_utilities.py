def chi_square_test(original_model, neighbor_model):
    """Perform chi-square difference test between two models"""
    chi2_diff = neighbor_model["chi2"] - original_model["chi2"]
    df_diff = neighbor_model["df"] - original_model["df"]

    # Get the p-value from the chi-square distribution
    p_value = 1 - chi2.cdf(chi2_diff, df_diff)

    return p_value


def likelihood_ratio_test(nested_log_likelihood, full_log_likelihood, df_diff):
    """
    Perform a Likelihood Ratio Test (LRT) to compare two nested models using their log-likelihoods.

    Parameters:
    - nested_log_likelihood: Log-likelihood of the nested (simpler) model.
    - full_log_likelihood: Log-likelihood of the full (more complex) model.
    - df_diff: Difference in degrees of freedom between the two models.

    Returns:
    - chi2_stat: The chi-square statistic for the LRT.
    - p_value: The p-value for the test.
    """
    # Ensure degrees of freedom difference is positive
    if df_diff <= 0:
        raise ValueError(
            "df_diff must be greater than 0 for LRT between full and nested models."
        )

    # Ensure that logL2 >= logL1
    if full_log_likelihood < nested_log_likelihood:
        raise ValueError(
            "logL2 should be greater than or equal to logL1. The full model should have a higher log-likelihood. Models are probably swapped or not nested."
        )

    # Calculate the likelihood ratio statistic (Δχ² = 2 * (logL2 - logL1))
    chi2_stat = 2 * (full_log_likelihood - nested_log_likelihood)

    # Compute the p-value using the chi-square distribution
    p_value = 1 - chi2.cdf(chi2_stat, df_diff)

    return chi2_stat, p_value


def chi2_model_fit_significantly_different(original_model, neighbor_model, alpha=0.05):
    """Use chi-square test to determine if the edge should be directed or undirected"""
    p_value = chi_square_test(original_model, neighbor_model)

    if p_value < alpha:
        # The test is significant, keep the edge directed
        return True
    else:
        # The test is not significant, make the edge undirected
        return False


def vuong_test(logL1, logL2, n_params1, n_params2, n_samples):
    """
    Perform the Vuong test to compare two non-nested models with the same degrees of freedom.
    This test gives a p-value based on the log-likelihoods, even if df_diff = 0.

    Parameters:
    ----------
    logL1 : float
        Log-likelihood of model 1.
    logL2 : float
        Log-likelihood of model 2.
    n_params1 : int
        Number of parameters in model 1.
    n_params2 : int
        Number of parameters in model 2.
    n_samples : int
        Number of observations or samples in the dataset.

    Returns:
    -------
    dict
        A dictionary containing:
            - 'z_stat': The z-statistic for the Vuong test.
            - 'p_value': The p-value associated with the z-statistic.
            - 'better_model': Indicates which model fits better ("Model 1", "Model 2", or "Indeterminate").
            - 'logL1': Log-likelihood of Model 1.
            - 'logL2': Log-likelihood of Model 2.
    """
    # Compute the log-likelihood ratio
    logL_diff = logL2 - logL1

    # Calculate the variance of the log-likelihood ratio (correcting for number of parameters)
    correction = (n_params2 - n_params1) / n_samples
    var_logL = np.abs(logL_diff) / n_samples

    # Compute the z-statistic
    z_stat = (logL_diff - correction) / np.sqrt(var_logL)

    # Compute the two-sided p-value from the z-statistic
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    # Determine which model is better based on the z-statistic
    if z_stat > 1.96:
        better_model = "Model 2"
    elif z_stat < -1.96:
        better_model = "Model 1"
    else:
        better_model = "Indeterminate (neither model is significantly better)"

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "better_model": better_model,
        "logL1": logL1,
        "logL2": logL2,
    }


def vuong_test(logL_per_obs_model1, logL_per_obs_model2):
    """
    Perform the Vuong test to compare two non-nested models using per-observation log-likelihoods.

    Parameters:
    ----------
    logL_per_obs_model1 : array-like
        Per-observation log-likelihoods of Model 1.
    logL_per_obs_model2 : array-like
        Per-observation log-likelihoods of Model 2.

    Returns:
    -------
    dict
        A dictionary containing:
            - 'z_stat': The z-statistic for the Vuong test.
            - 'p_value': The p-value associated with the z-statistic.
            - 'better_model': Indicates which model fits better ("Model 1", "Model 2", or "Indeterminate").
    """
    # Compute the difference in log-likelihoods per observation
    d = logL_per_obs_model1 - logL_per_obs_model2
    n = len(d)

    # Compute the mean and standard deviation of the differences
    mean_d = np.mean(d)
    sd_d = np.std(d, ddof=1)  # Use ddof=1 for sample standard deviation

    # Compute the z-statistic
    z_stat = mean_d / (sd_d / np.sqrt(n))

    # Compute the two-sided p-value from the z-statistic
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    # Determine which model is better based on the z-statistic
    if z_stat > 1.96:
        better_model = "Model 1"
    elif z_stat < -1.96:
        better_model = "Model 2"
    else:
        better_model = "Indeterminate (neither model is significantly better)"

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "better_model": better_model,
    }


import numpy as np
from scipy.stats import norm, chi2


def adjusted_vuong_test(
    logL_per_obs_model1, logL_per_obs_model2, n_params1, n_params2, adjustment="BIC"
):
    """
    Perform the Adjusted Vuong test to compare two non-nested models using per-observation log-likelihoods.

    Parameters:
    ----------
    logL_per_obs_model1 : array-like
        Per-observation log-likelihoods of Model 1.
    logL_per_obs_model2 : array-like
        Per-observation log-likelihoods of Model 2.
    n_params1 : int
        Number of parameters in Model 1.
    n_params2 : int
        Number of parameters in Model 2.
    adjustment : str, optional
        Type of adjustment to apply ('AIC', 'BIC', or 'none'). Default is 'BIC'.

    Returns:
    -------
    dict
        A dictionary containing:
            - 'z_stat': The adjusted z-statistic for the Vuong test.
            - 'p_value': The p-value associated with the z-statistic.
            - 'better_model': Indicates which model fits better ("Model 1", "Model 2", or "Indeterminate").
            - 'adjustment': The type of adjustment used.
    """
    # Compute the difference in log-likelihoods per observation
    d = logL_per_obs_model1 - logL_per_obs_model2
    n = len(d)

    # Compute the mean and standard deviation of the differences
    mean_d = np.mean(d)
    sd_d = np.std(d, ddof=1)  # Sample standard deviation

    # Adjustment for model complexity
    if adjustment.lower() == "aic":
        penalty = (n_params1 - n_params2) / n
    elif adjustment.lower() == "bic":
        penalty = ((n_params1 - n_params2) * np.log(n)) / (2 * n)
    elif adjustment.lower() == "none":
        penalty = 0
    else:
        raise ValueError("Invalid adjustment type. Choose 'AIC', 'BIC', or 'none'.")

    # Adjust the mean difference
    adjusted_mean_d = mean_d - penalty

    # Compute the adjusted z-statistic
    z_stat = adjusted_mean_d / (sd_d / np.sqrt(n))

    # Compute the two-sided p-value from the z-statistic
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    # Determine which model is better based on the z-statistic
    significance_level = (
        1.96  # Corresponds to a 5% significance level for a two-tailed test
    )
    if z_stat > significance_level:
        better_model = "Model 1"
    elif z_stat < -significance_level:
        better_model = "Model 2"
    else:
        better_model = "Indeterminate (neither model is significantly better)"

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "better_model": better_model,
        "adjustment": adjustment,
    }


def vuong_model_fit_significantly_different(model_a, model_b, alpha=0.05):
    """
    Use the log-likelihood ratio test to determine if the two models fit are significantly different.
    :param model_a:
    :param model_b:
    :param alpha:
    :return:
    """
    vuong_output = adjusted_vuong_test(
        model_a["log_likelihoods"],
        model_b["log_likelihoods"],
        model_a["npar"],
        model_b["npar"],
        adjustment="BIC",
    )

    p_value = vuong_output["p_value"]
    if p_value < alpha:
        # The test is significant, one model fits significantly better
        return True
    else:
        # The test is not significant, the models are not significantly different
        return False


def nested_loglikelihood_model_fit_significantly_different(
    full_model, nested_model, alpha=0.05
):
    """
    Use the log-likelihood ratio test to determine if the two models fit are significantly different.
    :param full_model:
    :param nested_model:
    :param alpha:
    :return:
    """
    # TODO Probably should be other tests to determine if the models are nested
    # Which model is nested ?
    if full_model["npar"] < nested_model["npar"]:
        # Assume original model is nested
        nested_model = full_model
        full_model = nested_model

    p_value = likelihood_ratio_test(
        nested_model["log_likelihood"],
        full_model["log_likelihood"],
        nested_model["df"] - full_model["df"],
    )[1]
    if p_value < alpha:
        # The test is significant, one model fits significantly better
        return True
    else:
        # The test is not significant, the models are not significantly different
        return False
