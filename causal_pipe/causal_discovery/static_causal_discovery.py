# Import necessary libraries
import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased.lingam import DirectLiNGAM, VARLiNGAM, ICALiNGAM, RCD
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder


import pandas as pd
import numpy as np
import warnings
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from causal_pipe.utilities.graph_utilities import graph_with_coefficient_to_pydot


def mi_to_pearson(mi):
    """
    Convert mutual information to an approximate Pearson correlation magnitude.

    :param mi: Mutual information value or array of values.
    :return: Approximate Pearson correlation magnitude(s).
    """
    return np.sqrt(1 - np.exp(-2 * mi))


def prepare_data_for_causal_discovery(
    data,
    handle_missing="drop",
    encode_categorical=True,
    scale_data=True,
    verbose=True,
    keep_only_correlated_with=None,
    filter_method="mutual_info",
    filter_threshold=0.1,
):
    """
    Prepares the data for causal discovery by handling missing values, encoding categorical variables,
    and scaling numerical variables.

    Parameters:
    - data (pd.DataFrame): The input data.
    - handle_missing (str): How to handle missing values ('drop' or 'impute').
    - encode_categorical (bool): Whether to encode categorical variables.
    - scale_data (bool): Whether to scale numerical variables.
    - verbose (bool): Whether to print processing steps.
    - keep_only_correlated_with (list): List of columns to use to filter uncorrelated columns.
    - filter_method (str): Method to use for filtering uncorrelated columns ('pearson', 'mutual_info', 'lasso').
    - filter_threshold (float): Threshold for filtering uncorrelated columns.

    Returns:
    - data_prepared (pd.DataFrame): The prepared data.
    """
    data = data.copy()

    # Step 1: Handle missing values
    if verbose:
        print("Handling missing values...")
    if handle_missing == "drop":
        data.dropna(inplace=True)
        if verbose:
            print(
                f"Dropped rows with missing values. Data now has {data.shape[0]} rows."
            )
    elif handle_missing == "impute":
        data.fillna(data.mean(), inplace=True)
        if verbose:
            print("Imputed missing values with column means.")
    elif handle_missing == "error":
        if data.isnull().values.any():
            raise ValueError("Missing values found in the data.")
        print("No missing values found in the data.")
    else:
        raise ValueError("handle_missing must be 'drop' or 'impute'.")

    # Step 2: Encode categorical variables
    if encode_categorical:
        if verbose:
            print("Encoding categorical variables...")
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            if verbose:
                print(f"Encoded {col}.")

    # Step 3: Verify data types
    if verbose:
        print("Verifying data types...")
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
            axis=1, how="any", inplace=True
        )  # Drop columns that couldn't be converted
    else:
        if verbose:
            print("All variables are numeric.")

    # Step 4: Scale numerical variables
    if scale_data:
        if verbose:
            print("Scaling numerical variables...")
        scaler = StandardScaler()
        data[numeric_vars] = scaler.fit_transform(data[numeric_vars])
        if verbose:
            print("Data has been scaled.")

    # Step 5: Keep only columns correlated with the specified columns
    if keep_only_correlated_with is not None and filter_method is None:
        warnings.warn(
            f"Cannot filter variables with keep_only_correlated_with!=None and filter_method=None, skipping variable filtering"
        )
    elif keep_only_correlated_with is not None:
        initial_columns = data.columns
        if verbose:
            print("Filtering out uncorrelated columns...")
        if filter_method in ["pearson", "spearman"]:
            corr_matrix = data.corr(method=filter_method)
            correlated_cols = (
                corr_matrix[
                    corr_matrix[keep_only_correlated_with].abs() > filter_threshold
                ]
                .dropna(axis=0, how="all")
                .index.tolist()
            )
            uncorrelated_cols = [
                col for col in data.columns if col not in correlated_cols
            ]
            data.drop(uncorrelated_cols, axis=1, inplace=True)
            if verbose:
                print(
                    f"Feature selection using {filter_method} correlation is complete."
                )
        elif filter_method == "mutual_info":  # Mutual information
            selected_features = set(keep_only_correlated_with)
            X = data.drop(columns=keep_only_correlated_with)
            y_cols = keep_only_correlated_with

            for y_col in y_cols:
                y = data[y_col]
                # Determine if y is continuous or discrete
                # if y.nunique() <= 10:
                #     y = y.astype("category")
                #     # Treat y as discrete
                #     mi = mutual_info_classif(
                #         X, y, discrete_features="auto", random_state=0
                #     )
                # else:
                # Treat y as continuous
                mi = mutual_info_regression(
                    X, y, discrete_features="auto", random_state=0
                )
                mi_series = pd.Series(mi, index=X.columns)
                # Select features with MI greater than threshold
                selected = mi_series[mi_series > filter_threshold].index.tolist()
                selected_features.update(selected)
                if verbose:
                    print(f"Selected features for {y_col}: {selected}")

            # Keep only selected features
            data = data[list(selected_features)]
            if verbose:
                print("Feature selection using Mutual Information is complete.")
        elif filter_method == "lasso":  # Lasso regression
            # Step 5: Feature selection using Lasso regression
            if verbose:
                print("Selecting features using Lasso regression...")
            from sklearn.linear_model import LassoCV

            selected_features = set(keep_only_correlated_with)
            X = data.drop(columns=keep_only_correlated_with)
            y_cols = keep_only_correlated_with

            for y_col in y_cols:
                y = data[y_col]
                # Handle the case where y is constant (Lasso cannot handle this)
                if y.nunique() <= 1:
                    if verbose:
                        print(f"Skipping {y_col} as it has only one unique value.")
                    continue
                # Fit LassoCV to find the best alpha
                lasso = LassoCV(cv=5, random_state=0)
                lasso.fit(X, y)
                # Get features with non-zero coefficients
                non_zero_coef_indices = np.where(lasso.coef_ != 0)[0]
                selected_features.update(X.columns[non_zero_coef_indices])
                if verbose:
                    print(
                        f"Selected features for {y_col}: {X.columns[non_zero_coef_indices].tolist()}"
                    )

            # Keep only selected features
            data = data[list(selected_features)]
            if verbose:
                print("Feature selection using Lasso is complete.")
        else:
            raise ValueError(
                "filter_method must be 'pearson', 'mutual_info', or 'lasso'."
            )
        if verbose:
            uncorrelated_cols = [
                col for col in initial_columns if col not in data.columns
            ]
            print(f"Uncorrelated columns have been removed : {uncorrelated_cols}.")

    if verbose:
        print("Data preparation is complete.")

    return data


def perform_data_validity_checks(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Performs statistical tests to assess the suitability of the data for causal discovery.

    Parameters:
    - data (pd.DataFrame): The prepared data.
    - verbose (bool): Whether to print test results.

    Returns:
    - test_results (dict): A dictionary containing test results.
    """
    test_results = {}

    # 1. Check for sufficient sample size
    n_samples, n_variables = data.shape
    test_results["sample_size"] = {
        "n_samples": n_samples,
        "n_variables": n_variables,
        "adequate": n_samples >= n_variables * 5,
    }
    if n_samples < n_variables * 5:
        warnings.warn(
            f"Sample size ({n_samples}) may be too small for the number of variables ({n_variables}). "
            f"Consider increasing the sample size or reducing the number of variables."
        )
    else:
        if verbose:
            print(
                f"Sample size ({n_samples}) is adequate for the number of variables ({n_variables})."
            )

    # 2. Check for missing values
    missing_total = data.isnull().sum().sum()
    missing_percentage = (missing_total / data.size) * 100
    test_results["missing_values"] = {
        "total_missing": missing_total,
        "percentage_missing": missing_percentage,
        "has_missing": missing_total > 0,
    }
    if missing_total > 0:
        warnings.warn(
            f"Data contains {missing_total} missing values ({missing_percentage:.2f}%). "
            "Consider handling missing data before proceeding."
        )
    else:
        if verbose:
            print("No missing values detected in the data.")

    # Missing values per column
    missing_per_column = data.isnull().mean() * 100
    test_results["missing_values_per_column"] = missing_per_column.to_dict()
    if verbose:
        print("Missing values per column:")
        print(missing_per_column)

    # 3. Check for constant columns (zero variance)
    constant_columns = data.columns[data.nunique() <= 1].tolist()
    test_results["constant_columns"] = constant_columns
    if constant_columns:
        warnings.warn(
            f"The following columns have zero variance (constant values): {constant_columns}. "
            "Consider removing them as they do not provide any information."
        )
    else:
        if verbose:
            print("No constant columns detected.")

    # 4. Multicollinearity Check using Variance Inflation Factor (VIF)
    # VIF requires no missing values and all numerical columns
    if test_results["missing_values"]["has_missing"]:
        warnings.warn(
            "Missing values detected. VIF calculation requires complete data. "
            "Consider handling missing values before performing multicollinearity checks."
        )
    else:
        numerical_data = data.select_dtypes(include=[np.number])
        if numerical_data.shape[1] < 2:
            test_results["multicollinearity"] = (
                "Not enough numerical variables to calculate VIF."
            )
            if verbose:
                print(
                    "Not enough numerical variables to perform multicollinearity check."
                )
        else:
            vif_data = pd.DataFrame()
            vif_data["feature"] = list(numerical_data.columns)
            vif_data["VIF"] = [
                variance_inflation_factor(numerical_data.values, i)
                for i in range(numerical_data.shape[1])
            ]
            test_results["multicollinearity"] = vif_data.to_dict(orient="list")
            high_vif = vif_data[vif_data["VIF"] > 5]
            if not high_vif.empty:
                warnings.warn(
                    f"High multicollinearity detected in the following variables:\n{high_vif}"
                )
            else:
                if verbose:
                    print("No multicollinearity issues detected based on VIF.")

            if verbose:
                print("Variance Inflation Factor (VIF) for numerical variables:")
                print(vif_data)

    # 5. Normality Check using Shapiro-Wilk test
    normality_results = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        stat, p_value = stats.shapiro(data[column].dropna())
        normality_results[column] = {
            "statistic": stat,
            "p_value": p_value,
            "normal": p_value > 0.05,
        }
        if p_value <= 0.05:
            warnings.warn(
                f"Variable '{column}' does not follow a normal distribution (p-value={p_value:.4f})."
            )
    test_results["normality"] = normality_results
    if verbose:
        print("Normality test results (Shapiro-Wilk):")
        for col, res in normality_results.items():
            status = "Normal" if res["normal"] else "Not Normal"
            print(f"  {col}: {status} (p-value={res['p_value']:.4f})")

    # 6. Outliers Detection using Z-score
    outliers = {}
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    for column in numerical_columns:
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        outlier_indices = data.index[z_scores > 3].tolist()
        outliers[column] = outlier_indices
        if len(outlier_indices) > 0:
            warnings.warn(
                f"Variable '{column}' has {len(outlier_indices)} outliers based on Z-score > 3."
            )
    test_results["outliers"] = outliers
    if verbose:
        print("Outliers detected (indices where |Z-score| > 3):")
        for col, indices in outliers.items():
            if indices:
                print(f"  {col}: {indices}")
            else:
                print(f"  {col}: No outliers detected.")

    return test_results


def get_graph_from_result(result, labels=None):
    """
    Extracts and returns a graph from the result of a causal discovery algorithm.

    Parameters:
    result (dict): The result dictionary containing information about the graph, such as adjacency matrix or edges.

    Returns:
    GeneralGraph: An instance of the GeneralGraph containing the causal structure.
    """
    if "adjacency_matrix" in result:
        adj_matrix = result["adjacency_matrix"]
        if labels is not None:
            nodes = [GraphNode(labels[i]) for i in range(len(labels))]
        else:
            nodes = [GraphNode(f"X{i}") for i in range(adj_matrix.shape[0])]
        graph = GeneralGraph(nodes)
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] != 0:
                    graph.add_directed_edge(nodes[i], nodes[j])
        return graph
    elif "adjacency_matrices" in result:  # For VAR-LiNGAM which has multiple matrices
        adj_matrices = result["adjacency_matrices"]
        if labels is not None:
            nodes = [GraphNode(labels[i]) for i in range(len(labels))]
        else:
            nodes = [GraphNode(f"X{i}") for i in range(adj_matrices[0].shape[0])]
        graph = GeneralGraph(nodes)
        for matrix in adj_matrices:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] != 0:
                        graph.add_directed_edge(nodes[i], nodes[j])
        return graph
    elif "G" in result:
        return result["G"]
    else:
        raise ValueError(
            "Result dictionary does not contain recognizable graph information."
        )


def run_causal_discovery(data, method, **kwargs):
    """
    Runs the specified causal discovery method on the provided data.

    Parameters:
    method (str): The name of the causal discovery algorithm to run.
    data (DataFrame): The input data for causal discovery.
    kwargs: Additional arguments for the causal discovery method.

    Returns:
    dict: A result dictionary containing causal structure and other relevant outputs.
    """
    parameters = kwargs.get("parameters", {})
    score_func = parameters.get("score_func", "local_score_marginal_general")
    if "kfold" in parameters and parameters["kfold"] > 1:
        score_func = "local_score_CV_general"
        if "lambda" not in parameters:
            parameters["lambda"] = 1
    if method == "pc":
        causal_graph = pc(data.values, show_progress=True, **kwargs)
        result = {"G": causal_graph.G, "causal_graph": causal_graph}
    elif method == "fci":
        # Run FCI algorithm
        g, edges = fci(data.values, **kwargs)
        result = {"G": g, "edges": edges}
    elif method == "ges":
        # Run GES algorithm
        if score_func == "local_score_marginal_general":
            score_func = "local_score_BIC"
        result = ges(data.values, score_func=score_func, **kwargs)
    elif method == "cdnod":
        # Run CD-NOD algorithm
        c_indx = kwargs.get(
            "c_indx", -1
        )  # Default to the last column as changing factor index
        causal_graph = cdnod(data.values, c_indx, **kwargs)
        result = {"G": causal_graph.G, "causal_graph": causal_graph}
    elif method == "icalingam":
        # Run ICA-LiNGAM algorithm
        model = ICALiNGAM(**kwargs)
        model.fit(data.values)
        result = {
            "causal_order": model.causal_order_,
            "adjacency_matrix": model.adjacency_matrix_,
        }
        result["G"] = get_graph_from_result(result, labels=data.columns)
    elif method == "directlingam":
        # Run DirectLiNGAM algorithm
        model = DirectLiNGAM(**kwargs)
        model.fit(data.values)
        result = {
            "causal_order": model.causal_order_,
            "adjacency_matrix": model.adjacency_matrix_,
        }
        result["G"] = get_graph_from_result(result, labels=data.columns)
    elif method == "varlingam":
        # Run VAR-LiNGAM algorithm
        model = VARLiNGAM(**kwargs)
        model.fit(data.values)
        result = {
            "causal_order": model.causal_order_,
            "adjacency_matrices": model.adjacency_matrices_,
        }
        result["G"] = get_graph_from_result(result, labels=data.columns)
    elif method == "rcd":
        # Run RCD algorithm
        model = RCD(**kwargs)
        model.fit(data.values)
        result = {
            "adjacency_matrix": model.adjacency_matrix_,
            "ancestors_list": model.ancestors_list_,
        }
        result["G"] = get_graph_from_result(result, labels=data.columns)
    elif method == "grasp":
        # Run GES algorithm
        if score_func == "local_score_marginal_general":
            score_func = "local_score_BIC"  # weird bug sometimes
        # Run GRaSP algorithm
        result = grasp(data.values, score_func=score_func, **kwargs)
        result = {"G": result}
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def visualize_causal_graph(
    result, labels=None, title=None, output_path=None, show=True
):
    """
    Visualizes the causal graph.

    Parameters:
    - result: The result object from the causal discovery algorithm.
    - labels (list): Optional list of labels for the nodes.
    - filename (str): Optional filename to save the graph as an image.
    """
    title = title or "Causal Graph"

    if not show and output_path is None:
        raise ValueError("Please specify an output path or set show=True.")
    visualize_graph(
        result["G"], title=title, show=show, output_path=output_path, labels=labels
    )


def visualize_graph(
    graph, edges=None, title=None, show=True, output_path=None, labels=None
):
    """
    Visualizes the causal graph.

    Parameters:
    - graph (GeneralGraph): The causal graph to visualize.
    - title (str): The title of the graph.
    - show (bool): Whether to display the graph.
    - output_path (str): The path to save the graph image.
    - labels (list): Optional list of labels for the nodes
    """
    pyd = graph_with_coefficient_to_pydot(graph, edges=edges, labels=labels, dpi=300)
    tmp_png = pyd.create_png()
    sio = io.BytesIO()
    sio.write(tmp_png)
    sio.seek(0)
    img = mpimg.imread(sio)
    plt.figure(figsize=(16, 16), dpi=300)
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"{title} saved as {output_path}.")
    if show:
        plt.show()


def cross_validate_causal_discovery(
    data, method="fci", k_folds=5, random_state=None, **kwargs
):
    """
    Performs cross-validation for causal discovery algorithms.

    Parameters:
    - data (pd.DataFrame): The prepared data.
    - method (str): The causal discovery method ('fci', 'ges', 'grasp', etc.).
    - cv (int): Number of cross-validation folds.
    - random_state (int): Random state for reproducibility.
    - **kwargs: Additional parameters for the causal discovery algorithms.

    Returns:
    - cv_results (list): List of results from each cross-validation fold.
    """
    if method in ["ges", "grasp"]:
        # Set 'kfold' in parameters for methods that accept it
        parameters = kwargs.get("parameters", {})
        parameters["kfold"] = (
            k_folds  # Set k-fold parameter for internal cross-validation
        )
        kwargs["parameters"] = parameters

        # Call run_causal_discovery directly since the method handles cross-validation internally
        result = run_causal_discovery(data, method, **kwargs)
        return [result]

    else:
        # For methods that do not have internal cross-validation, perform external cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        cv_results = []

        fold = 1
        for train_index, test_index in kf.split(data):
            print(f"Cross-validation fold {fold}")
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            result = run_causal_discovery(X_train, method, **kwargs)
            cv_results.append(result)
            fold += 1

        return cv_results


# Example usage
if __name__ == "__main__":
    # Load your data into a pandas DataFrame
    # For demonstration purposes, we'll create synthetic data
    np.random.seed(42)
    num_samples = 500
    data = pd.DataFrame(
        {
            "X": np.random.normal(size=num_samples),
            "Y": np.random.normal(size=num_samples),
            "Z": np.random.normal(size=num_samples),
        }
    )
    data["W"] = (
        data["X"] + data["Y"] + np.random.normal(size=num_samples)
    )  # W depends on X and Y

    # Prepare the data
    data_prepared = prepare_data_for_causal_discovery(
        data, handle_missing="drop", encode_categorical=False, scale_data=True
    )

    # Perform statistical tests
    test_results = perform_data_validity_checks(data_prepared)

    # Run causal discovery algorithm
    method = "icalingam"
    result = run_causal_discovery(data_prepared, method=method)

    # Visualize the causal graph
    labels = data_prepared.columns.tolist()
    visualize_causal_graph(
        result, title=f"{method.upper()} Result, no cross-validation"  # , labels=labels
    )

    # Perform cross-validation
    methods = ["ges", "fci", "grasp", "icalingam", "directlingam", "varlingam", "rcd"]
    for method in methods:
        print(f"Running cross-validation for {method}")
        cv_results = cross_validate_causal_discovery(
            data_prepared, method=method, k_folds=5
        )

        # Optionally, you can visualize graphs from each fold
        if len(cv_results) > 1:
            for idx, res in enumerate(cv_results):
                print(f"Fold {idx + 1}")
                visualize_causal_graph(
                    res,
                    labels=labels,
                    title=f"Fold {idx + 1} result for method {method}",
                )
        else:
            print("Cross-validation results:")
            visualize_causal_graph(
                cv_results[0],
                labels=labels,
                title=f"Cross-validation Result for method {method}",
            )
