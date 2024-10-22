import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
from typing import List, Tuple, Any


# Define a function to fit a mixed model
def fit_mixed_model(data: pd.DataFrame, formula: str, groups: str) -> MixedLMResults:
    """
    Fits a mixed linear model to the provided data.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataset containing the variables for the model.
    formula : str
        The model formula (e.g., 'y ~ x').
    groups : str
        The column name in `data` that defines the grouping structure.

    Returns:
    -------
    result : MixedLMResults
        The fitted mixed linear model result.
    """
    # Initialize the mixed linear model using the formula and grouping variable
    model = MixedLM.from_formula(formula, groups=groups, data=data)

    # Fit the model to the data
    result = model.fit()

    return result


# Define a function to extract parameters from the mixed model result
def extract_parameters(result: MixedLMResults) -> Tuple[pd.Series, pd.Series]:
    """
    Extracts parameter estimates and their standard errors from a mixed model result.

    Parameters:
    ----------
    result : MixedLMResults
        The fitted mixed linear model result.

    Returns:
    -------
    params : pd.Series
        The estimated coefficients for each parameter.
    bse : pd.Series
        The standard errors of the estimated coefficients.
    """
    return result.params, result.bse


# Define a function to pool results using Rubin's rules
def pool_mixed_model_results(results: List[Tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """
    Pools multiple mixed model results using Rubin's rules for multiple imputation.

    Parameters:
    ----------
    results : List[Tuple[pd.Series, pd.Series]]
        A list of tuples, each containing parameter estimates and standard errors from different imputed datasets.

    Returns:
    -------
    pooled_results : pd.DataFrame
        A DataFrame containing the pooled parameter estimates and their pooled standard errors.
    """
    # Extract coefficients and standard errors from each model result
    estimates_list = [res[0] for res in results]  # Collecting coefficients (pd.Series)
    ses_list = [res[1] for res in results]  # Collecting standard errors (pd.Series)

    # Combine estimates into a DataFrame for easy handling
    estimates_df = pd.DataFrame(estimates_list)
    ses_df = pd.DataFrame(ses_list)

    # Number of imputations
    m = len(results)

    # Calculate pooled estimates (mean of estimates across imputations)
    pooled_estimates = estimates_df.mean(axis=0)

    # Calculate within-imputation variance (mean of squared standard errors)
    within_var = (ses_df ** 2).mean(axis=0)

    # Calculate between-imputation variance (variance of estimates)
    between_var = estimates_df.var(axis=0, ddof=1)

    # Total variance: within + (1 + 1/m) * between
    total_var = within_var + (1 + 1 / m) * between_var

    # Pooled standard errors
    pooled_ses = np.sqrt(total_var)

    # Create a DataFrame with parameter names, pooled estimates, and standard errors
    pooled_results = pd.DataFrame({
        "Parameter": pooled_estimates.index,
        "Pooled Estimate": pooled_estimates.values,
        "Pooled Standard Error": pooled_ses.values,
    })

    return pooled_results


if __name__ == "__main__":
    # Example usage with imputed datasets
    # Assuming you have 3 datasets after multiple imputation
    data1 = pd.DataFrame({
        "y": np.random.randn(1000),
        "x": np.random.randn(1000),
        "group": np.random.choice(["A", "B"], size=1000),
    })
    data2 = pd.DataFrame({
        "y": np.random.randn(1000),
        "x": np.random.randn(1000),
        "group": np.random.choice(["A", "B"], size=1000),
    })
    data3 = pd.DataFrame({
        "y": np.random.randn(1000),
        "x": np.random.randn(1000),
        "group": np.random.choice(["A", "B"], size=1000),
    })

    # Fit mixed models to each dataset
    results = []
    for data in [data1, data2, data3]:
        # Fit the mixed model using 'y ~ x' as the formula and 'group' as the grouping variable
        result = fit_mixed_model(data, "y ~ x", groups="group")

        # Extract parameter estimates and standard errors
        params, ses = extract_parameters(result)

        # Append the results as a tuple to the results list
        results.append((params, ses))

    # Pool the results from all imputed datasets
    pooled_estimates = pool_mixed_model_results(results)

    # Display the pooled parameter estimates and their standard errors
    print("Pooled Estimates:")
    print(pooled_estimates)
