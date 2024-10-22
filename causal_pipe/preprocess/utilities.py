import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df, covariates):
    """
    Helper function to calculate VIF for given covariates.
    """
    X = df[covariates].assign(constant=1).dropna()
    vif_df = pd.DataFrame()
    vif_df["Variable"] = covariates
    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(covariates))
    ]
    return vif_df


def check_vif(df, covariates, vif_threshold=10, drop_high_vif=True, should_plot=True):
    """
    Calculate Variance Inflation Factor (VIF) for a given set of covariates and explore multicollinearity in detail.

    :param df: pandas DataFrame containing the data
    :param covariates: List of covariate column names
    :param vif_threshold: VIF threshold to drop variables (default=10)
    :param drop_high_vif: Whether to drop variables with high VIF values and recalculate (default=True)
    :return: DataFrame with VIF values for each covariate, and optionally drop high VIF variables
    """
    initial_covariates = covariates.copy()
    # Step 1: Initial VIF calculation
    vif_df = calculate_vif(df, covariates)
    if should_plot:
        print("Initial VIF values:\n", vif_df)

    # Step 2: Correlation matrix to identify highly correlated variables
    correlation_matrix = df[covariates].corr()

    # Visualizing the correlation matrix using Seaborn heatmap
    if should_plot:
        print("\nCorrelation matrix:\n", correlation_matrix)
        plt.figure(figsize=(20, 20))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        plt.show()

    # Step 3: Iteratively drop variables with high VIF
    if drop_high_vif:
        high_vif_vars = vif_df[vif_df["VIF"] > vif_threshold]["Variable"].tolist()
        while len(high_vif_vars) > 0:
            # Drop the variable with the highest VIF value
            highest_vif = vif_df["VIF"].max()
            max_vif_var = vif_df.loc[vif_df["VIF"].idxmax(), "Variable"]
            print(f"\nDropping '{max_vif_var}' with VIF = {vif_df['VIF'].max()}\n")
            covariates.remove(max_vif_var)

            # Recalculate VIF after dropping the variable
            vif_df = calculate_vif(df, covariates)
            if should_plot:
                print("Updated VIF values:\n", vif_df)

            # Check for any remaining high VIF values
            high_vif_vars = vif_df[vif_df["VIF"] > vif_threshold]["Variable"].tolist()
    removed_covariates = [c for c in initial_covariates if c not in covariates]
    print(f"\nRemoved covariates: {removed_covariates}")
    kept_columns = [c for c in df.columns if c not in removed_covariates]
    df_with_low_vif = df[kept_columns]

    return vif_df, df_with_low_vif


def ensure_data_types(
    df,
    categorical_cols,
    float_cols,
    group_cols=None,
    cat_to_codes=False,
    standardize=False,
):
    """
    Prepares the DataFrame by converting specified columns to categorical and float types.

    :param df: pandas DataFrame containing the data
    :param categorical_cols: List of column names that should be converted to categorical
    :param float_cols: List of column names that should be converted to float
    :param group_cols: List of column names that should be converted to categorical and then to integer codes
    :param cat_to_codes: Whether to convert categorical columns to integer codes
    :param standardize: Whether to z-normalize the float columns
    :return: The DataFrame with corrected data types
    """
    # Convert specified columns to categorical
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            if cat_to_codes:
                na_indices = df[col].isna()
                df[col] = df[col].cat.codes
                df.loc[na_indices, col] = np.nan
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Convert specified columns to float
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float").astype(
                np.float64
            )
            if standardize:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Convert group columns to categorical, then map them to integers
    if group_cols is not None:
        for col in group_cols:
            if col in df.columns:
                na_indices = df[col].isna()
                df[col] = (
                    df[col].astype("category").cat.codes
                )  # Convert to categorical, then to integer codes
                df.loc[na_indices, col] = np.nan
            else:
                raise ValueError(f"Column '{col}' not found in DataFrame")

    return df
