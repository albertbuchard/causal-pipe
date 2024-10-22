import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from statsmodels.imputation.mice import MICEData

def convert_to_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to appropriate nullable extension types.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns converted to nullable types.
    """
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            # Convert integer columns to nullable integer type
            df[col] = df[col].astype(pd.Int64Dtype())
        elif pd.api.types.is_float_dtype(df[col]):
            # Convert float columns to nullable float type
            df[col] = df[col].astype(pd.Float64Dtype())
        elif pd.api.types.is_bool_dtype(df[col]):
            # Convert boolean columns to nullable boolean type
            df[col] = df[col].astype(pd.BooleanDtype())
        elif pd.api.types.is_string_dtype(df[col]) and not isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            # Convert string columns to nullable string type
            df[col] = df[col].astype(pd.StringDtype())
    return df

def create_code_to_label_map(
    categorical_series: pd.Series, start_from: int = 0
) -> Dict[int, str]:
    """
    Create a mapping from the category codes to their labels for a pandas Categorical series.

    Parameters:
    ----------
    categorical_series : pd.Series
        A pandas Series with categorical data.
    start_from : int, optional
        The starting integer for category codes, by default 0.

    Returns:
    -------
    Dict[int, str]
        A dictionary mapping category codes to labels.

    Raises:
    ------
    ValueError
        If the provided series is not of a categorical dtype.
    """
    if not isinstance(categorical_series.dtype, pd.CategoricalDtype):
        raise ValueError("The provided series is not of a categorical dtype.")

    # Create a mapping from category codes to labels
    code_to_label_map = {
        (start_from + code): label
        for code, label in enumerate(categorical_series.cat.categories)
    }
    return code_to_label_map

def pandas_to_r_df_with_factors(
    df: pd.DataFrame
) -> Tuple[Any, Dict[str, Dict[int, str]]]:
    """
    Convert a pandas DataFrame to an R data.frame, ensuring that categorical columns are converted to factors.

    Parameters:
    ----------
    df : pd.DataFrame
        The input pandas DataFrame.

    Returns:
    -------
    Tuple[Any, Dict[str, Dict[int, str]]]
        A tuple containing the R data.frame and a dictionary of original category mappings.
    """
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()

    # Ensure that categorical columns are of dtype 'category'
    df = df.copy()

    # Replace all np.nan or null with pd.NA
    df = df.where(pd.notnull(df), pd.NA)

    all_categorical_cols = []
    original_categories = {}

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            all_categorical_cols.append(col)
            original_categories[col] = create_code_to_label_map(df[col], start_from=1)

    # Convert pandas DataFrame to R data.frame
    r_df = pandas2ri.py2rpy(df)

    return r_df, original_categories

def r_mice_impute(
    r_df: Any,
    m: int = 5,
    maxit: int = 5,
    method: Optional[List[str]] = None,
    seed: Optional[int] = None,
    original_categories: Optional[Dict[str, Dict[int, str]]] = None,
) -> List[pd.DataFrame]:
    """
    Perform multiple imputation using R's mice package on the R dataframe.

    Parameters:
    ----------
    r_df : Any
        The R dataframe to impute.
    m : int, optional
        Number of multiple imputations, by default 5.
    maxit : int, optional
        Number of iterations, by default 5.
    method : Optional[List[str]], optional
        A list specifying the imputation method to be used for each column in data, by default None.
    seed : Optional[int], optional
        Seed for random number generator in R, by default None.
    original_categories : Optional[Dict[str, Dict[int, str]]], optional
        Mapping of original categories for categorical columns, by default None.

    Returns:
    -------
    List[pd.DataFrame]
        A list of imputed pandas DataFrames.
    """
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import StrVector
    import rpy2.robjects.packages as rpackages

    pandas2ri.activate()

    # Import necessary R packages
    utils = rpackages.importr("utils")
    base = rpackages.importr("base")

    # Import mice package
    if not rpackages.isinstalled("mice"):
        utils.install_packages("mice")
    mice = importr("mice")

    # Set seed in R
    if seed is not None:
        r["set.seed"](seed)

    # Call mice function
    # If method is provided, it should be a list of methods for each column
    if method is not None:
        method_vector = StrVector(method)
        imp = mice.mice(r_df, m=m, method=method_vector, maxit=maxit, printFlag=False)
    else:
        imp = mice.mice(r_df, m=m, maxit=maxit, printFlag=False)

    # Extract imputed datasets
    imputed_dfs = []
    for i in range(1, m + 1):
        # Get the i-th completed dataset
        complete_data = mice.complete_mids(imp, action=i)
        # Convert R dataframe to pandas DataFrame
        df_imputed = pandas2ri.rpy2py(complete_data)
        if original_categories is not None:
            for col in original_categories:
                df_imputed[col] = pd.Categorical(
                    df_imputed[col]
                ).rename_categories(original_categories[col])
        imputed_dfs.append(df_imputed)

    return imputed_dfs

def perform_multiple_imputation(
    scores_df: pd.DataFrame,
    impute_cols: List[str],
    full_obs_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    method: str = "mice",
    r_mice: bool = False,
) -> List[pd.DataFrame]:
    """
    Perform multiple imputation on the impute_cols, using full_obs_cols as predictors.

    Parameters:
    ----------
    scores_df : pd.DataFrame
        The pandas DataFrame containing the data.
    impute_cols : List[str]
        List of columns to impute.
    full_obs_cols : Optional[List[str]], optional
        List of columns without missing data (assumed fully observed), by default None.
    categorical_cols : Optional[List[str]], optional
        List of categorical columns to impute, by default None.
    method : str, optional
        Imputation method, options are "mice", "simple", or any method accepted by R mice, by default "mice".
    r_mice : bool, optional
        If True, use R mice package for imputation, by default False.

    Returns:
    -------
    List[pd.DataFrame]
        A list of DataFrames with imputed values in the impute_cols; full_obs_cols are untouched.

    Raises:
    ------
    ValueError
        If impute_cols is not provided or empty.
        If full_obs_cols contain missing values.
        If there is a mismatch in categorical columns when converting to R data.frame.
    """
    if impute_cols is None or not impute_cols:
        raise ValueError("impute_cols must be provided and not empty.")

    # Check for missing values in full_obs_cols
    if full_obs_cols is None:
        scores_df_clean = scores_df
        full_obs_cols = []
    else:
        na_counts = scores_df[full_obs_cols].isna().sum(axis=0)
        if na_counts.any():
            raise ValueError("full_obs_cols contain missing values.")

        # Drop rows where any full_obs_cols have missing values
        scores_df_clean = scores_df.dropna(subset=full_obs_cols)

    # Create a copy of the DataFrame to avoid modifying the original
    imputed_df = scores_df_clean.copy()

    if method == "mice" and not r_mice and categorical_cols is not None:
        warnings.warn("Using R mice package for imputation of categorical columns.")
        r_mice = True

    if r_mice:
        # Use R mice package for imputation
        from rpy2.robjects.packages import importr
        import rpy2.robjects.packages as rpackages

        utils = rpackages.importr("utils")

        # Ensure the R 'mice' package is installed
        if not rpackages.isinstalled("mice"):
            utils.install_packages("mice")

        # Combine full_obs_cols and impute_cols
        cols_to_use = full_obs_cols + impute_cols
        data_for_imputation = imputed_df[cols_to_use]

        # Convert categorical columns to 'category' dtype
        if categorical_cols is not None:
            for col in categorical_cols:
                data_for_imputation[col] = data_for_imputation[col].astype("category")

        # Convert pandas DataFrame to R data.frame with factors
        r_df, original_categories = pandas_to_r_df_with_factors(data_for_imputation)
        if set(original_categories.keys()) != set(categorical_cols):
            raise ValueError(
                "Mismatch in categorical columns when converting to R data.frame."
            )

        # Prepare method vector
        if method == "mice" or method is None:
            method = None  # Let R mice decide default methods
        elif method == "auto":
            # Create method list matching the columns in r_df
            method_list = []
            for col in cols_to_use:
                if col in full_obs_cols:
                    method_list.append("")  # No imputation for fully observed columns
                elif col in categorical_cols:
                    # Use appropriate method for categorical data
                    method_list.append("polyreg")  # Adjust method as needed
                else:
                    # Use method for numerical data
                    method_list.append("pmm")  # Adjust method as needed
            method = method_list

        # Call r_mice_impute
        imputed_dfs = r_mice_impute(
            r_df, method=method, original_categories=original_categories
        )

        # Ensure the imputed DataFrame has the correct data types
        for i, idf in enumerate(imputed_dfs):
            for col in categorical_cols or []:
                if not isinstance(idf[col].dtype, pd.CategoricalDtype):
                    raise ValueError(
                        f"Column '{col}' is not of categorical dtype after imputation. Something went wrong."
                    )

        # Return the list of imputed DataFrames
        return imputed_dfs

    else:
        # Continue with existing method using Python's MICE implementation
        if categorical_cols is not None:
            if method == "mice":
                for col in categorical_cols:
                    imputed_df[col] = imputed_df[col].astype("category")
                imputation_data = imputed_df[full_obs_cols + impute_cols]
                mice_data = MICEData(imputation_data)
                for _ in range(10):  # Number of iterations, adjust as needed
                    mice_data.update_all()
                imputed_df = mice_data.data.copy()

            elif method == "simple":
                # Use SimpleImputer for categorical data
                cat_imputer = SimpleImputer(strategy="most_frequent")
                imputed_df[categorical_cols] = cat_imputer.fit_transform(
                    imputed_df[categorical_cols]
                )

                # For numerical columns, use IterativeImputer
                numerical_cols = [
                    col for col in impute_cols if col not in categorical_cols
                ]
                if numerical_cols:
                    num_imputer = IterativeImputer(
                        random_state=0,
                        max_iter=10,
                        sample_posterior=True,
                        keep_empty_features=True,
                    )
                    imputation_data = imputed_df[full_obs_cols + numerical_cols]
                    imputed_data = num_imputer.fit_transform(imputation_data)
                    imputed_df[numerical_cols] = imputed_data[:, len(full_obs_cols) :]

        else:
            # If no categorical columns, use IterativeImputer for numerical data
            imputation_data = imputed_df[full_obs_cols + impute_cols]
            imputer = IterativeImputer(
                random_state=0,
                max_iter=10,
                sample_posterior=True,
                keep_empty_features=True,
            )
            imputed_data = imputer.fit_transform(imputation_data)
            imputed_df[impute_cols] = imputed_data[:, len(full_obs_cols) :]

        # Return the DataFrame with imputed columns
        return [imputed_df]

def generate_simulated_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate simulated data with missing values for demonstration purposes.

    Parameters:
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    random_state : int, optional
        Seed for random number generator, by default 42.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing simulated numerical and categorical data with missing values.
    """
    np.random.seed(random_state)
    # Simulate numerical data
    age = np.random.randint(18, 80, size=n_samples)
    income = np.random.normal(50000, 15000, size=n_samples)
    # Simulate categorical data
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    occupation = np.random.choice(
        ["Engineer", "Doctor", "Artist", "Lawyer"], size=n_samples
    )

    # Create DataFrame
    data = pd.DataFrame(
        {"Age": age, "Income": income, "Gender": gender, "Occupation": occupation}
    )

    # Introduce missing values at random
    for col in ["Income", "Gender", "Occupation"]:
        data.loc[data.sample(frac=0.2, random_state=random_state).index, col] = pd.NA

    return data

if __name__ == "__main__":
    # Generate the simulated data
    simulated_data = generate_simulated_data()

    print("Simulated Data with Missing Values:")
    print(simulated_data.head(20))

    print("\nMissing Values in Simulated Data:")
    print(simulated_data.isnull().sum())

    # Define columns
    full_obs_cols = ["Age"]
    impute_cols = ["Income", "Gender", "Occupation"]
    categorical_cols = ["Gender", "Occupation"]

    # Perform imputation using R's mice package
    imputed_data_r = perform_multiple_imputation(
        scores_df=simulated_data,
        impute_cols=impute_cols,
        full_obs_cols=full_obs_cols,
        categorical_cols=categorical_cols,
        method="mice",
        r_mice=True,
    )

    print("\nImputed Data using R's mice package:")
    for i, df_imputed in enumerate(imputed_data_r, start=1):
        print(f"\nImputed Dataset {i}:")
        print(df_imputed.head(20))

    # Check that there is no missing data
    for i, df_imputed in enumerate(imputed_data_r, start=1):
        print(f"\nMissing Values in Imputed Data {i}:")
        print(df_imputed.isnull().sum())
