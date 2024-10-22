import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler


# Function 0: Prepare Data


def prepare_data_for_fa(
    data,
    method="error",
    missing_threshold=0.05,
    variance_threshold=0.0,
    corr_threshold=0.9,
    normality_threshold=0.05,
    sample_size_check=True,
    standardize=True,
    verbose=True,
    id_column=None,
):
    """
    Preprocesses the data for factor analysis by performing the following steps:
    1. Handles missing values.
    2. Removes variables with zero variance.
    3. Checks and addresses multicollinearity.
    4. Standardizes the data.
    5. Optionally assesses sample size adequacy.
    6. Assesses and addresses non-normality.
    7. Verifies data types.

    Parameters:
    - data (pd.DataFrame): The input data for factor analysis.
    - method (str): Method to handle missing values. 'drop' to remove rows with missing values,
      'impute' to fill with mean values.
    - missing_threshold (float): Maximum allowed proportion of missing values per variable.
      Variables with missing proportion higher than this will be removed.
    - variance_threshold (float): Threshold below which variables with low variance will be removed.
    - corr_threshold (float): Threshold above which variables are considered highly correlated.
    - normality_threshold (float): p-value threshold for the Shapiro-Wilk test to assess normality.
    - sample_size_check (bool): If True, checks if the sample size is adequate for FA.
    - standardize (bool): If True, standardizes the data.
    - verbose (bool): If True, prints out the steps and findings.
    - id_column (str): The name of the column containing the unique identifier for each observation.

    Returns:
    - data_prepared (pd.DataFrame): The preprocessed data ready for factor analysis.
    """
    data = data.copy()
    if id_column is not None:
        data.set_index(id_column, inplace=True)

    # Step 1: Handle missing values
    if method == "error" and data.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in the data.")

    if verbose:
        print("Step 1: Handling missing values...")
    # Remove variables with too many missing values
    missing_proportion = data.isnull().mean()
    vars_to_remove = missing_proportion[
        missing_proportion > missing_threshold
    ].index.tolist()
    if vars_to_remove:
        if verbose:
            print(
                f"Variables with more than {missing_threshold * 100}% missing values:"
            )
            print(vars_to_remove)
        data.drop(columns=vars_to_remove, inplace=True)

    if method == "drop":
        data.dropna(inplace=True)
        if verbose:
            print(
                f"Dropped rows with missing values. Data now has {data.shape[0]} rows."
            )
    elif method == "impute":
        data.fillna(data.mean(), inplace=True)
        if verbose:
            print("Imputed missing values with column means.")
    else:
        raise ValueError("Method must be 'drop' or 'impute'.")

    # Step 2: Remove variables with zero or near-zero variance
    if verbose:
        print("\nStep 2: Removing variables with near-zero variance...")
    variance = data.var()
    vars_to_remove = variance[variance <= variance_threshold].index.tolist()
    if vars_to_remove:
        if verbose:
            print(f"Variables with variance <= {variance_threshold}:")
            print(vars_to_remove)
        data.drop(columns=vars_to_remove, inplace=True)
    else:
        if verbose:
            print("No variables with near-zero variance found.")

    # Step 3: Check and address multicollinearity
    if verbose:
        print("\nStep 3: Checking for multicollinearity...")
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    vars_to_remove = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > corr_threshold)
    ]
    if vars_to_remove:
        if verbose:
            print(
                f"Variables to remove due to high correlation (threshold = {corr_threshold}):"
            )
            print(vars_to_remove)
        data.drop(columns=vars_to_remove, inplace=True)
    else:
        if verbose:
            print("No highly correlated variables found.")

    # Step 4: Verify data types
    if verbose:
        print("\nStep 4: Verifying data types...")
    numeric_vars = data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_vars = [col for col in data.columns if col not in numeric_vars]
    if non_numeric_vars:
        if verbose:
            print("Converting non-numeric variables to numeric (if possible):")
            print(non_numeric_vars)
        data[non_numeric_vars] = data[non_numeric_vars].apply(
            pd.to_numeric, errors="coerce"
        )
        data.dropna(
            axis=1, how="all", inplace=True
        )  # Drop columns that couldn't be converted
    else:
        if verbose:
            print("All variables are numeric.")

    # Step 5: Assess sample size adequacy
    if sample_size_check:
        if verbose:
            print("\nStep 5: Assessing sample size adequacy...")
        n_samples, n_variables = data.shape
        if n_samples < n_variables * 5:
            warnings.warn(
                f"Sample size ({n_samples}) may be too small for the number of variables ({n_variables})."
            )
        else:
            if verbose:
                print(
                    f"Sample size ({n_samples}) is adequate for the number of variables ({n_variables})."
                )

    # Step 6: Assess normality of variables
    if verbose:
        print("\nStep 6: Assessing normality of variables...")
    non_normal_vars = []
    for column in data.columns:
        stat, p = shapiro(data[column])
        if p < normality_threshold:
            non_normal_vars.append(column)
            if verbose:
                print(f"{column} does not follow a normal distribution (p = {p:.4f}).")
    if non_normal_vars:
        if verbose:
            print("Consider transforming these variables or using robust methods.")
    else:
        if verbose:
            print("All variables appear to be normally distributed.")

    # Step 7: Standardize the data
    if standardize:
        if verbose:
            print("\nStep 7: Standardizing the data...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_prepared = pd.DataFrame(
            data_scaled, columns=data.columns, index=data.index
        )
        if verbose:
            print("Data has been standardized.")
    else:
        data_prepared = data.copy()
        if verbose:
            print("Data has not been standardized.")

    # Reassessment of normality after standardization
    if verbose:
        print("\nReassessing normality of standardized variables...")
    non_normal_vars = []
    for column in data_prepared.columns:
        stat, p = shapiro(data_prepared[column])
        if p < normality_threshold:
            non_normal_vars.append(column)
            if verbose:
                print(f"{column} does not follow a normal distribution (p = {p:.4f}).")
    if non_normal_vars:
        if verbose:
            print("Consider using robust methods.")
    else:
        if verbose:
            print(
                "All variables appear to be normally distributed after standardization."
            )

    # Final Step: Return the prepared data
    if verbose:
        print("\nData preparation is complete. Data is ready for factor analysis.")

    if id_column is not None:
        data_prepared.reset_index(inplace=True)

    return data_prepared


# Function 1: Check Suitability for Factor Analysis
def check_suitability(data):
    """
    Checks if the data is suitable for factor analysis using Bartlett's Test and the KMO test.

    Parameters:
    data (DataFrame): The item-level data for factor analysis.

    Returns:
    bool: True if data is suitable, False otherwise.
    """
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity,
        calculate_kmo,
    )

    chi_square_value, p_value = calculate_bartlett_sphericity(data)
    kmo_all, kmo_model = calculate_kmo(data)

    print(f"Bartlett's Test p-value: {p_value}")
    print(f"KMO Test Score: {kmo_model}")

    if p_value < 0.05 and kmo_model > 0.6:
        print("Data is suitable for factor analysis.")
        return True
    else:
        print("Data not suitable for factor analysis.")
        return False


# Function 2: Determine Number of Factors
def determine_number_of_factors(data):
    """
    Determines the number of factors to retain using eigenvalues and a scree plot.

    Parameters:
    data (DataFrame): The item-level data for factor analysis.

    Returns:
    int: Suggested number of factors to retain.
    """
    from factor_analyzer import FactorAnalyzer

    fa = FactorAnalyzer(rotation=None)
    fa.fit(data)
    eigenvalues, _ = fa.get_eigenvalues()

    # Create scree plot
    plt.figure(figsize=(8, 6))
    plt.scatter(range(1, len(eigenvalues) + 1), eigenvalues, color="blue")
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, color="blue")
    plt.title("Scree Plot")
    plt.xlabel("Factor Number")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.show()

    # Determine number of factors where eigenvalue > 1
    n_factors = sum(eigenvalues > 1)
    print(f"Number of factors with eigenvalue > 1: {n_factors}")

    return n_factors


# Function 3: Perform Exploratory Factor Analysis
def perform_exploratory_fa(data, n_factors):
    """
    Performs exploratory factor analysis and returns factor loadings and scores.

    Parameters:
    data (DataFrame): The item-level data for factor analysis.
    n_factors (int): Number of factors to retain.

    Returns:
    DataFrame: Factor loadings.
    DataFrame: Factor scores.
    """
    from factor_analyzer import FactorAnalyzer

    # Perform factor analysis with oblique rotation
    fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin")
    fa.fit(data)

    # Get factor loadings
    loadings = pd.DataFrame(fa.loadings_, index=data.columns)
    print("\nFactor Loadings:")
    print(loadings)

    # Compute factor scores
    factor_scores = fa.transform(data)
    factor_scores_df = pd.DataFrame(
        factor_scores,
        columns=[f"Factor_{i+1}" for i in range(n_factors)],
        index=data.index,
    )

    return loadings, factor_scores_df


# Function 4: Get Model Structure as String
def get_model_structure_to_string(loadings, threshold=0.4):
    """
    Converts factor loadings into a model string for CFA.

    Parameters:
    loadings (DataFrame): Factor loadings from EFA.
    threshold (float): Minimum absolute loading value to include an item.

    Returns:
    str: Model specification string for CFA.
    """
    all_variables = loadings.index
    factor_items = {}
    for factor in loadings.columns:
        items = loadings.index[loadings[factor].abs() >= threshold].tolist()
        factor_items[factor] = items

    missing_variables = set(all_variables) - set(
        item for items in factor_items.values() for item in items
    )

    model_lines = []
    for i, (factor, items) in enumerate(factor_items.items(), start=1):
        if items:
            items_str = " + ".join(items)
            line = f"Factor{i} =~ {items_str}"
            model_lines.append(line)

    model_string = "\n".join(model_lines)
    print("\nModel Structure for CFA:")
    print(model_string)
    return model_string, missing_variables


def perform_cfa_and_check_invariance(data, model_string, group_variable):
    """
    Performs CFA and tests for measurement invariance across groups using lavaan via rpy2.

    Parameters:
    data (DataFrame): The full dataset including the group variable.
    model_string (str): The model specification string for CFA in lavaan syntax.
    group_variable (str): The name of the column indicating group membership.

    Returns:
    None
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects.lib.ggplot2 as ggplot2

    # Activate automatic pandas conversion
    pandas2ri.activate()

    # Import necessary R packages
    utils = rpackages.importr("utils")
    base = rpackages.importr("base")

    # Ensure the lavaan package is installed
    if not rpackages.isinstalled("lavaan"):
        utils.install_packages("lavaan")
    lavaan = rpackages.importr("lavaan")

    # Convert the pandas DataFrame to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Assign the data to an R variable
    ro.globalenv["data"] = r_data

    # Convert the model string to an R string
    ro.globalenv["model_string"] = model_string

    # Set up the group variable
    ro.globalenv["group_variable"] = group_variable

    # Load the lavaan package in R
    ro.r("library(lavaan)")

    # Configural Invariance Model (no constraints)
    ro.r(
        """
        # Fit the configural invariance model
        fit1 <- cfa(model = model_string, data = data, group = group_variable)
        summary_fit1 <- summary(fit1, fit.measures = TRUE)
    """
    )

    # Print the configural invariance model fit indices
    print("\nConfigural Invariance Model Fit Indices:")
    print(ro.r("summary_fit1"))

    # Weak (Metric) Invariance Model (constrain loadings to be equal)
    ro.r(
        """
        # Fit the weak (metric) invariance model
        fit2 <- cfa(model = model_string, data = data, group = group_variable, group.equal = c("loadings"))
        summary_fit2 <- summary(fit2, fit.measures = TRUE)
    """
    )

    # Print the weak invariance model fit indices
    print("\nWeak (Metric) Invariance Model Fit Indices:")
    print(ro.r("summary_fit2"))

    # Strong (Scalar) Invariance Model (constrain loadings and intercepts to be equal)
    ro.r(
        """
        # Fit the strong (scalar) invariance model
        fit3 <- cfa(model = model_string, data = data, group = group_variable, group.equal = c("loadings", "intercepts"))
        summary_fit3 <- summary(fit3, fit.measures = TRUE)
    """
    )

    # Print the strong invariance model fit indices
    print("\nStrong (Scalar) Invariance Model Fit Indices:")
    print(ro.r("summary_fit3"))

    # Model Comparison Tests
    ro.r(
        """
        # Perform chi-square difference tests
        comparison <- lavTestLRT(fit1, fit2, fit3)
    """
    )

    # Print the comparison
    print("\nComparison of Models:")
    print(ro.r("comparison"))


def get_factor_scores_from_cfa(data, model_string, factor_names=None):
    """
    Performs CFA using the specified model_string and returns the factor scores with meaningful factor names.

    Parameters:
    data (DataFrame): The dataset including all variables needed for the CFA.
    model_string (str): The model specification string for CFA in lavaan syntax.

    Returns:
    DataFrame: A DataFrame containing the factor scores for each participant with renamed factors.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages
    import pandas as pd

    # Activate automatic pandas conversion
    pandas2ri.activate()

    # Import necessary R packages
    utils = rpackages.importr("utils")
    base = rpackages.importr("base")

    # Ensure the lavaan package is installed
    if not rpackages.isinstalled("lavaan"):
        utils.install_packages("lavaan")
    lavaan = rpackages.importr("lavaan")

    # Convert the pandas DataFrame to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Assign the data to an R variable
    ro.globalenv["data"] = r_data

    # Assign the model string to an R variable
    ro.globalenv["model_string"] = model_string

    # Load the lavaan package in R
    ro.r("library(lavaan)")

    # Fit the CFA model
    ro.r(
        """
        fit <- cfa(model = model_string, data = data, meanstructure = TRUE)
        summary_fit <- summary(fit, fit.measures = TRUE)
    """
    )

    # Print the CFA summary
    print("\nCFA Model Fit Summary:")
    print(ro.r("summary_fit"))

    # Get the factor scores
    ro.r(
        """
        factor_scores <- lavPredict(fit)
    """
    )

    # Convert factor scores from R to pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        factor_scores_df = ro.conversion.rpy2py(ro.r("as.data.frame(factor_scores)"))

    # Ensure that the factor names in the DataFrame match the keys in factor_names
    if factor_names is not None:
        # Get the current factor names from the DataFrame
        current_factor_names = factor_scores_df.columns.tolist()

        # Create a mapping of current factor names to meaningful names
        rename_mapping = {
            current_name: factor_names.get(current_name, current_name)
            for current_name in current_factor_names
        }

        # Rename the columns
        factor_scores_df.rename(columns=rename_mapping, inplace=True)

    # Reset index to align with the original data
    factor_scores_df.reset_index(drop=True, inplace=True)
    factor_scores_df.index = data.index

    return factor_scores_df
