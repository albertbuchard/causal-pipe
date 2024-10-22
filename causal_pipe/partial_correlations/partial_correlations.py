from itertools import combinations

import numpy as np
import pandas as pd
from bcsl.fci import fci_orient_edges_from_graph_node_sepsets
from bcsl.graph_utils import get_undirected_graph_from_skeleton
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.KCI.KCI import KCI_UInd, KCI_CInd
from causallearn.utils.cit import CIT
from npeet_plus import mi_pvalue, mi
from scipy import stats
from scipy.stats import spearmanr
from sklearn.covariance import GraphicalLassoCV
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


def compute_correlations(df):
    """
    Compute the Pearson correlation matrix from a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input data.

    Returns:
    - corr_matrix (np.ndarray): The Pearson correlation matrix.
    """
    return df.corr().values


def find_smallest_sepset(
    df,
    i,
    j,
    start_sepset,
    method,
    alpha,
    epsilon,
    verbose,
    max_condition_set_size,
    **kwargs,
) -> set:
    """
    Find the smallest separation set (sepset) for variables i and j.

    Parameters:
    - df (pd.DataFrame): The input data.
    - i (int): Index of the first variable.
    - j (int): Index of the second variable.
    - start_sepset (set): Initial set of potential conditioning variables.
    - method (str): Correlation method used ('glasso', etc.).
    - alpha (float): Significance level for tests.
    - correction_method (str): Method for conditional independence testing.
    - epsilon (float): Threshold for minimal correlation.
    - verbose (bool): Whether to print progress messages.
    - max_condition_set_size (int or None): Max size of conditioning sets.
    - **kwargs: Additional keyword arguments.

    Returns:
    - smallest_sepset (set): The minimal conditioning set found.
    """
    # Initialize the smallest sepset as the full start_sepset
    smallest_sepset = set(start_sepset)

    # If no conditioning variables are needed, return empty set
    if not start_sepset:
        return smallest_sepset

    # Iterate over conditioning set sizes from 0 up to max_condition_set_size
    max_size = (
        max_condition_set_size
        if max_condition_set_size is not None
        else len(start_sepset)
    )
    for size in range(0, max_size + 1):
        if verbose:
            print(
                f"Testing conditioning sets of size {size} for variables {i} and {j}."
            )

        # Generate all possible combinations of the current size
        for S in combinations(start_sepset, size):
            S_set = set(S)
            if verbose:
                print(f"Testing conditioning set: {S_set}")

            # Extract the relevant data
            X = df.iloc[:, i].values
            Y = df.iloc[:, j].values
            Z = df.iloc[:, list(S_set)].values if S_set else None

            # Perform the partial correlation test
            r, p = partial_corr_test(X, Y, Z, conditional_indepence_method=method)

            # Check for conditional independence
            if p >= alpha and abs(r) < epsilon:
                if verbose:
                    print(
                        f"Conditional independence detected for ({i}, {j}) given {S_set} with r={r:.4f}, p={p:.4f}"
                    )
                return S_set  # Found the smallest sepset

    # If no sepset found that renders conditional independence, return the full start_sepset
    if verbose:
        print(
            f"No conditioning set found that renders ({i}, {j}) conditionally independent. Using full sepset."
        )
    return smallest_sepset


def compute_partial_correlations_corrected(
    df,
    method="glasso",
    alpha=0.05,
    correction_method="pearson",
    verbose=False,
    epsilon=1e-6,
    max_condition_set_size=4,
    refine_sepsets=False,
    **kwargs,
):
    """
    Compute the partial correlation matrix from a pandas DataFrame,
    detecting and correcting for edges due to collider biases.

    Parameters:
    - df (pd.DataFrame): The input data.
    - method (str): The method to compute the precision matrix ('glasso' supported).
    - alpha (float): Significance level for statistical tests.
    - correction_method (str): The correlation method to use to correct for collider bias ('pearson', 'spearman', 'kci').
    - verbose (bool): Whether to print progress messages.
    - epsilon (float): Smallest value considered a non-zero partial correlation.
    - max_condition_set_size (int): Maximum size of conditioning sets to consider during sepset identification.
    - refine_sepsets (bool): Whether to refine sepsets using a permutation test.
    - **kwargs: Additional keyword arguments for the estimator.

    Returns:
    - corrected_partial_corr_matrix (np.ndarray): The partial correlation matrix corrected for collider bias.
    """

    # Step 0: Compute the pairwise correlation matrix
    if verbose:
        print("Computing pairwise correlations...")
    corr_matrix, _ = compute_significant_pairwise_correlations(
        df, method, alpha, verbose, epsilon, **kwargs
    )

    # Step 1: Compute the partial correlation matrix
    if verbose:
        print("Computing partial correlations...")
    partial_corr_matrix = compute_partial_correlations(
        df, method, verbose=verbose, **kwargs
    )

    # Step 1b: Create the sepsets dictionary with all lost edges
    sepsets = {}
    n_vars = partial_corr_matrix.shape[0]
    all_nodes = set(range(n_vars))

    if verbose:
        print("Identifying sepsets for zero partial correlations...")
        outer_iter = tqdm(range(n_vars), desc="Sepset For Variables")
    else:
        outer_iter = range(n_vars)

    for i in outer_iter:
        for j in range(i + 1, n_vars):
            sepsets[(i, j)] = set()
            sepsets[(j, i)] = set()
            if corr_matrix[i, j] != 0:
                X = df.iloc[:, i]
                Y = df.iloc[:, j]
                Z = df.values[:, [k for k in range(n_vars) if k != i and k != j]]
                r, p_value_partial_corr = partial_corr_test(
                    X, Y, Z, conditional_indepence_method="fisherz"
                )
                if p_value_partial_corr >= alpha:
                    # Define the initial sepset as all other nodes except i and j
                    start_sepset = set(all_nodes) - {i, j}

                    if verbose:
                        print(f"\nFinding sepset for pair ({i}, {j})...")

                    # Find the smallest sepset for this pair
                    smallest_sepset = start_sepset
                    if refine_sepsets:
                        smallest_sepset = find_smallest_sepset(
                            df,
                            i=i,
                            j=j,
                            start_sepset=start_sepset,
                            method=(
                                "pearson"
                                if correction_method == "glasso"
                                else "fisherz"
                            ),
                            alpha=alpha,
                            epsilon=epsilon,
                            verbose=verbose,
                            max_condition_set_size=max_condition_set_size,
                            **kwargs,
                        )

                    # Store the sepset
                    sepsets[(i, j)] = smallest_sepset
                    sepsets[(j, i)] = smallest_sepset  # Ensure symmetry

                    # Remove the edge from the partial correlation matrix
                    partial_corr_matrix[i, j] = 0
                    partial_corr_matrix[j, i] = 0

    # Initialize corrected partial correlation matrix
    print("Correcting for collider bias...")
    corrected_partial_corr_matrix = partial_corr_matrix.copy()

    # Step 2: Detect and correct for colliders
    n_vars = df.shape[1]
    variables = df.columns.tolist()

    # Iterate over all combinations of triplets
    tqdm_bar = tqdm(combinations(range(n_vars), 3), desc="Detecting colliders")
    for i, j, k in tqdm_bar:
        if corrected_partial_corr_matrix[i, j] < epsilon:
            corrected_partial_corr_matrix[i, j] = 0
            corrected_partial_corr_matrix[j, i] = 0
            continue
        tqdm_bar.set_postfix(
            {"Edge": f"{variables[i]} - {variables[j]} | {variables[k]}"}
        )
        X = df.iloc[:, i]
        Y = df.iloc[:, j]
        Z = df.iloc[:, k]

        # Compute zero-order correlations
        if correction_method == "pearson":
            r_xy, p_xy = stats.pearsonr(X, Y)
        elif correction_method == "spearman":
            r_xy, p_xy = spearmanr(X, Y)
        elif correction_method == "kci":
            X = X.values.reshape(-1, 1)
            Y = Y.values.reshape(-1, 1)
            Z = Z.values.reshape(-1, 1)
            kci = KCI_UInd(null_ss=1000, approx=True)
            p_xy, r_xy = kci.compute_pvalue(data_x=X, data_y=Y)
        elif correction_method == "conditional_mi":
            X = X.values
            Y = Y.values
            Z = Z.values
            r_xy, p_xy, _ = mi_pvalue(X, Y, alpha=0.25)
        else:
            raise ValueError(
                "Invalid correlation method. Choose 'pearson', 'spearman', or 'kci'."
            )

        # Compute partial correlations controlling for Z
        r_xy_z, p_xy_z = partial_corr_test(
            X, Y, Z, conditional_indepence_method=correction_method
        )

        # Check for collider bias: If X and Y are independent (p > alpha)
        # but become dependent when conditioning on Z (p < alpha)
        if p_xy > alpha and p_xy_z < alpha:
            if verbose:
                print(
                    f"Collider detected: {variables[i]}, {variables[j]} | {variables[k]}"
                )
            # Spurious association due to collider at Z
            # Remove edge between X and Y in the partial correlation matrix
            corrected_partial_corr_matrix[i, j] = 0
            corrected_partial_corr_matrix[j, i] = 0

    return corrected_partial_corr_matrix, sepsets


def get_skeleton_from_adjacency_matrix(adjacency_matrix):
    undirected_adjacency = np.abs(adjacency_matrix) > 0
    np.fill_diagonal(undirected_adjacency, False)
    edges = np.argwhere(undirected_adjacency)
    edges = list(set([tuple(sorted(edge)) for edge in edges]))
    return edges


def fci_orient_edges_from_adjacency_and_sepsets(
    data,
    adjacency_matrix,
    sepsets,
    background_knowledge=None,
    node_names=None,
    independence_test_method="fisherz",
    verbose=False,
    alpha=0.05,
    max_path_length=3,
):
    if node_names is None:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
        else:
            node_names = [f"Var{i}" for i in range(adjacency_matrix.shape[0])]

    skeleton = get_skeleton_from_adjacency_matrix(adjacency_matrix)

    graph = get_undirected_graph_from_skeleton(skeleton=skeleton, node_names=node_names)
    nodes = graph.nodes

    independence_test_fn = CIT(data.values, method=independence_test_method)

    graph, edges = fci_orient_edges_from_graph_node_sepsets(
        data=data,
        graph=graph,
        nodes=nodes,
        sepsets=sepsets,
        background_knowledge=background_knowledge,
        independence_test_method=independence_test_fn,
        alpha=alpha,
        max_path_length=max_path_length,
        verbose=verbose,
    )

    return graph, edges


def partial_corr_test(
    X, Y, Z, conditional_indepence_method="pearson", Z_target=None, Z_source=None
):
    """
    Compute the partial correlation between X and Y, controlling for Z,
    and perform a statistical test.

    Parameters:
    - X, Y, Z: pd.Series or np.ndarray. The variables involved.
    - correlation_method (str): The correlation method to use ('pearson', 'spearman', 'kci', "conditional_mi").
    - Z_target: pd.Series or np.ndarray. Conditioning variables for the target variable if different from Z, Z will be used for the source variable.
    - Z_source: pd.Series or np.ndarray. Conditioning variables for the source variable if different from Z, Z will be used for the target variable.

    Returns:
    - r (float): Partial correlation coefficient.
    - p_value (float): Two-tailed p-value.
    :param Z_source:
    :param Z_target:
    """
    if isinstance(X, pd.Series):
        X = X.values
    if isinstance(Y, pd.Series):
        Y = Y.values
    if isinstance(Z, pd.Series):
        Z = Z.values
    if isinstance(Z_target, pd.Series):
        Z_target = Z_target.values
    if isinstance(Z_source, pd.Series):
        Z_source = Z_source.values
    # If Z is unidimensional, reshape it to a column vector
    if Z is not None and Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z_target is not None and Z_target.ndim == 1:
        Z_target = Z_target.reshape(-1, 1)
    if Z_source is not None and Z_source.ndim == 1:
        Z_source = Z_source.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if (
        conditional_indepence_method == "pearson"
        or conditional_indepence_method == "spearman"
    ):
        if Z is not None:
            Z_target = Z
            Z_source = Z
        if Z_source is None and Z_target is None:
            if conditional_indepence_method == "pearson":
                r, p_value = stats.pearsonr(X, Y)
            else:
                r, p_value = spearmanr(X, Y)
        else:
            # Regress X on Z
            if Z_source is None:
                residuals_x = X
            else:
                beta_xz = np.linalg.lstsq(Z_source, X, rcond=None)[0]
                residuals_x = X - Z_source @ beta_xz

            # Regress Y on Z or Z_target if provided
            if Z_target is None:
                residuals_y = Y
            else:
                beta_yz = np.linalg.lstsq(Z_target, Y, rcond=None)[0]
                residuals_y = Y - Z_target @ beta_yz

            # Compute correlation between residuals
            if conditional_indepence_method == "pearson":
                r, p_value = stats.pearsonr(
                    residuals_x.squeeze(), residuals_y.squeeze()
                )
            else:
                r, p_value = spearmanr(residuals_x, residuals_y)
    elif conditional_indepence_method == "fisherz":
        if Z_target is not None or Z_source is not None:
            raise ValueError(
                "Fisher Z test does not support different conditioning variables for source and target"
            )
        data_np = np.column_stack((X, Y))
        if Z is not None:
            data_np = np.column_stack((data_np, Z))
        cit = CIT(data=data_np, method="fisherz")
        p_value = cit(0, 1, [i for i in range(2, data_np.shape[1])])
        r = 1 if p_value <= 0.05 else 0
    elif conditional_indepence_method == "kci":
        if Z_target is not None or Z_source is not None:
            raise ValueError(
                "KCI test does not support different conditioning variables for source and target"
            )
        kci_test = KCI_CInd(nullss=5000, approx=True)
        # Perform the KCI test
        p_value, r = kci_test.compute_pvalue(data_x=X, data_y=Y, data_z=Z)
    elif conditional_indepence_method == "conditional_mi":
        r, p_value, _ = mi_pvalue(X, Y, Z, alpha=0.25)
    else:
        raise ValueError(
            "Invalid correlation method. Choose 'pearson', 'spearman', or 'kci', 'fisherz'."
        )

    return r, p_value


def compute_partial_correlation_kci(
    df: pd.DataFrame,
    null_ss: int = 5000,
    alpha: float = 0.05,
    known_graph: GeneralGraph = None,
    verbose: bool = False,
):
    """
    Compute the non-linear partial correlation matrix using the KCI_CInd test.

    Parameters:
    - df: pandas DataFrame with continuous variables.
    - null_ss: Number of samples in simulating the null distribution (default=5000).
    - alpha: Significance level for the hypothesis test (default=0.05).
    - known_graph: Known graph structure to use for conditional independence tests (default=None).
    - verbose: Whether to display progress messages (default=False).

    Returns:
    - stat_matrix: numpy array representing the test statistics matrix.
    - pvalue_matrix: numpy array representing the p-values matrix.
    - significance_matrix: numpy array indicating significant partial correlations (1 if significant, 0 otherwise).
    """
    n_vars = df.shape[1]
    columns = df.columns
    stat_matrix = np.zeros((n_vars, n_vars))
    pvalue_matrix = np.ones((n_vars, n_vars))
    significance_matrix = np.zeros((n_vars, n_vars))
    np.fill_diagonal(stat_matrix, 0.0)
    np.fill_diagonal(pvalue_matrix, 0.0)
    np.fill_diagonal(significance_matrix, 0.0)

    if verbose:
        iterator = tqdm(range(n_vars), desc="Variables")
    else:
        iterator = range(n_vars)

    for i in iterator:
        for j in range(i + 1, n_vars):
            if known_graph is not None:
                node_i = known_graph.nodes[i]
                node_j = known_graph.nodes[j]
                if not known_graph.is_adjacent_to(node_i, node_j):
                    continue
                _, Z_columns = get_parents_or_undirected(known_graph, i)
                Z_columns = [columns[col] for col in Z_columns if col != j]
            else:
                Z_columns = [col for k, col in enumerate(columns) if k != i and k != j]
            if Z_columns:
                Z = df[Z_columns].values
            else:
                Z = None

            X = df.iloc[:, i].values.reshape(-1, 1)
            Y = df.iloc[:, j].values.reshape(-1, 1)

            kci_test = KCI_CInd(nullss=null_ss, approx=True)
            # Perform the KCI test
            pvalue, test_stat = kci_test.compute_pvalue(data_x=X, data_y=Y, data_z=Z)
            # Store the test statistic and p-value
            stat_matrix[i, j] = test_stat
            stat_matrix[j, i] = test_stat
            pvalue_matrix[i, j] = pvalue
            pvalue_matrix[j, i] = pvalue
            # Determine significance based on alpha
            if pvalue < alpha:
                significance_matrix[i, j] = 1
                significance_matrix[j, i] = 1

    return stat_matrix, pvalue_matrix, significance_matrix


def get_parents_or_undirected(graph: GeneralGraph, node_idx: int):
    target_node = graph.nodes[node_idx]
    all_edges = graph.get_node_edges(target_node)
    children = graph.get_children(target_node)
    if children:
        all_edges = [
            edge
            for edge in all_edges
            if edge.node2 not in children and edge.node1 not in children
        ]
    predictors_nodes = [
        edge.node1 if edge.node1 != target_node else edge.node2 for edge in all_edges
    ]
    node_map = graph.node_map
    return [node for node in predictors_nodes], [
        node_map[node] for node in predictors_nodes
    ]


def compute_partial_correlations(
    df,
    method="glasso",
    graphical_lasso_args=None,
    verbose=False,
    known_graph: GeneralGraph = None,
) -> np.ndarray:
    """
    Compute the partial correlation matrix from a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input data.
    - method (str): The method to compute the partial correlation matrix ('glasso', 'pearson', 'spearman', 'kci', "conditional_mi").
    - graphical_lasso_args (dict): Additional arguments for GraphicalLassoCV.
    - verbose (bool): Whether to print progress messages.

    Returns:
    - partial_corr_matrix (np.ndarray): The partial correlation matrix.
    """

    n_vars = df.shape[1]
    partial_corr_matrix = np.zeros((n_vars, n_vars))

    if method == "glasso":
        if known_graph is not None:
            raise ValueError("glasso method does not support known graph")
        graphical_lasso_args = graphical_lasso_args or {}
        model = GraphicalLassoCV(**graphical_lasso_args)
        model.fit(df)
        precision_matrix = model.precision_

        # Convert precision matrix to partial correlations
        d = np.sqrt(np.diag(precision_matrix))
        partial_corr_matrix = -precision_matrix / np.outer(d, d)
        np.fill_diagonal(partial_corr_matrix, 1)
    elif method in ["pearson", "spearman"]:
        # Compute residuals using linear regression for pearson or spearman correlation
        residuals = pd.DataFrame(index=df.index, columns=df.columns)
        for source_idx, source_name in enumerate(df.columns):
            if known_graph is not None:
                _, predictor_node_indices = get_parents_or_undirected(
                    known_graph, source_idx
                )
                node_names = list(df.columns)
                predictors = [node_names[idx] for idx in predictor_node_indices]
                z_source = None
                for target_idx, target_name in enumerate(node_names):
                    if target_idx == source_idx or not known_graph.is_adjacent_to(
                        known_graph.nodes[source_idx], known_graph.nodes[target_idx]
                    ):
                        continue
                    if predictors:
                        predictors_without_target = [
                            col for col in predictors if col != target_name
                        ]
                        if predictors_without_target:
                            z_source = df[predictors_without_target].values

                    _, target_predictors_indices = get_parents_or_undirected(
                        known_graph, target_idx
                    )
                    target_predictors = [
                        node_names[idx]
                        for idx in target_predictors_indices
                        if idx != source_idx
                    ]
                    z_target = None
                    if target_predictors:
                        z_target = df[target_predictors].values

                    r, p = partial_corr_test(
                        X=df[source_name],
                        Y=df[target_name],
                        Z=None,
                        conditional_indepence_method=method,
                        Z_source=z_source,
                        Z_target=z_target,
                    )
                    partial_corr_matrix[source_idx, target_idx] = r
                    partial_corr_matrix[target_idx, source_idx] = r

            else:
                predictors = [col for col in df.columns if col != source_name]

                if not predictors:
                    residuals[source_name] = df[source_name]
                else:
                    predictors = df[predictors].values
                    beta = np.linalg.lstsq(predictors, df[source_name], rcond=None)[0]
                    residuals[source_name] = df[source_name] - predictors.dot(beta)

                # Compute correlations
                if method == "pearson":
                    partial_corr_matrix = residuals.corr(method="pearson").values
                else:
                    partial_corr_matrix = residuals.corr(method="spearman").values
    elif method in ["conditional_mi", "mi"]:
        columns = df.columns
        partial_corr_matrix = np.zeros((n_vars, n_vars))
        np.fill_diagonal(partial_corr_matrix, 1.0)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                x = df.iloc[:, i].values
                y = df.iloc[:, j].values
                if known_graph is not None:
                    if not known_graph.is_adjacent_to(
                        known_graph.nodes[i], known_graph.nodes[j]
                    ):
                        continue
                    # For i
                    _, predictor_node_indices = get_parents_or_undirected(
                        known_graph, i
                    )
                    z_columns = [
                        columns[idx] for idx in predictor_node_indices if idx != j
                    ]
                    # For j
                    _, target_predictors_indices = get_parents_or_undirected(
                        known_graph, j
                    )
                    z_columns += [
                        columns[idx] for idx in target_predictors_indices if idx != i
                    ]
                else:
                    z_columns = [
                        col for k, col in enumerate(columns) if k != i and k != j
                    ]

                z = None
                if z_columns:
                    z_columns = list(set(z_columns))
                    z = df[z_columns].values

                cmi = mi(x, y, z, k=3, alpha=0.25)
                if cmi < 0:
                    cmi = 0
                partial_corr_matrix[i, j] = cmi
                partial_corr_matrix[j, i] = cmi
    elif method == "kci":
        stat_matrix, pvalue_matrix, significance_matrix = (
            compute_partial_correlation_kci(
                df, verbose=verbose, known_graph=known_graph
            )
        )
        partial_corr_matrix = significance_matrix * stat_matrix
    else:
        raise ValueError(
            "Invalid method: choose 'glasso'', 'pearson', 'spearman', or 'kci'."
        )

    # Remove edges not present in the known graph
    if known_graph is not None:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                node_i = known_graph.nodes[i]
                node_j = known_graph.nodes[j]
                if not known_graph.is_adjacent_to(node_i, node_j):
                    partial_corr_matrix[i, j] = 0
                    partial_corr_matrix[j, i] = 0

    # Set the diagonal to 0
    np.fill_diagonal(partial_corr_matrix, 0)

    return partial_corr_matrix


def compute_significant_pairwise_correlations(
    df,
    method="pearson",
    alpha=0.05,
    verbose=False,
    epsilon=1e-6,
    **kwargs,
):
    """
    Compute the significant pairwise correlations from a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input data.
    - method (str): The method to compute the correlation. Supported methods:
        - 'pearson': Pearson correlation coefficient.
        - 'spearman': Spearman rank correlation.
        - 'kendall': Kendall Tau correlation.
        - 'mutual_info': Mutual Information.
        - 'kci': Conditional Mutual Information using KCI test.
        - 'conditional_mi': Conditional Mutual Information non-parametric.
    - alpha (float): Significance level for statistical tests.
    - verbose (bool): Whether to print progress messages.
    - epsilon (float): Minimum absolute correlation value to consider as non-zero.
    - **kwargs: Additional keyword arguments for specific methods.

    Returns:
    - correlation_matrix (np.ndarray): A symmetric matrix with significant correlations.
      Non-significant correlations are set to 0.
    - pvalue_matrix (np.ndarray): A symmetric matrix of p-values for each correlation.
    """
    n_vars = df.shape[1]
    columns = df.columns
    correlation_matrix = np.zeros((n_vars, n_vars))
    pvalue_matrix = np.ones((n_vars, n_vars))

    # Initialize the iterator with tqdm if verbose
    if verbose:
        iterator = tqdm(range(n_vars), desc="Computing Correlations")
    else:
        iterator = range(n_vars)

    if method == "glasso":
        method = "pearson"

    for i in iterator:
        for j in range(i + 1, n_vars):
            var1 = df.iloc[:, i].values
            var2 = df.iloc[:, j].values

            if method == "pearson":
                r, p = stats.pearsonr(var1, var2)
            elif method == "spearman":
                r, p = stats.spearmanr(var1, var2)
            elif method == "kendall":
                r, p = stats.kendalltau(var1, var2)
            elif method == "mutual_info":
                # mutual_info_regression expects 2D array for X
                mi = mutual_info_regression(var1.reshape(-1, 1), var2, **kwargs)
                r = mi[0]
                # Mutual Information does not provide a p-value directly
                # Here, we can approximate p-value via permutation or set to NaN
                p = np.nan  # Placeholder
            elif method == "kci":
                # Example using KCI_CInd
                kci_test = KCI_UInd(null_ss=1000, approx=True)
                var1 = var1.reshape(-1, 1)
                var2 = var2.reshape(-1, 1)
                p, r = kci_test.compute_pvalue(data_x=var1, data_y=var2)
            elif method == "conditional_mi":
                # Example using a non-parametric mutual information p-value
                try:
                    z = df.drop(columns=[columns[i], columns[j]]).values
                except KeyError:
                    z = None  # No conditioning variables
                r, p, _ = mi_pvalue(var1, var2, z, alpha=kwargs.get("alpha", 0.25))
            else:
                raise ValueError(
                    f"Invalid method: '{method}'. Supported methods are 'pearson', 'spearman', 'kendall', 'mutual_info', 'kci', 'conditional_mi'."
                )

            # Determine significance
            is_significant = False
            if method in [
                "pearson",
                "spearman",
                "kendall",
                "kci",
                "conditional_mi",
            ]:
                if p < alpha and abs(r) >= epsilon:
                    is_significant = True
            elif method == "mutual_info":
                # For mutual_info, p-value is not directly available
                # Users may need to handle significance externally
                # Here, we'll set significance based on epsilon only
                if r >= epsilon:
                    is_significant = True
                    p = np.nan  # Indicate p-value is not computed
                else:
                    p = 1.0  # Not significant

            if is_significant:
                correlation_matrix[i, j] = r
                correlation_matrix[j, i] = r
                pvalue_matrix[i, j] = p
                pvalue_matrix[j, i] = p
            else:
                # Non-significant correlations remain zero
                pass

    np.fill_diagonal(correlation_matrix, 1)
    return correlation_matrix, pvalue_matrix
