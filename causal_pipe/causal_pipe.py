import logging
import os
import traceback
import warnings
from typing import Optional, Dict, Any, Tuple, List, Set

import pandas as pd
from bcsl.bcsl import BCSL
from bcsl.fci import fci_orient_edges_from_graph_node_sepsets
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.FAS import fas
from causallearn.utils.cit import CIT

from causal_pipe.causal_discovery.static_causal_discovery import (
    prepare_data_for_causal_discovery,
    perform_data_validity_checks,
    visualize_graph,
)
from causal_pipe.imputation.imputation import perform_multiple_imputation
from causal_pipe.partial_correlations.partial_correlations import (
    compute_partial_correlations,
)
from causal_pipe.preprocess.utilities import ensure_data_types
from causal_pipe.sem.sem import (
    fit_sem_lavaan,
    search_best_graph_climber,
)
from causal_pipe.causal_discovery.fas_bootstrap import bootstrap_fas_edge_stability
from causal_pipe.causal_discovery.bootstrap_utils import make_graph
from causal_pipe.utilities.graph_utilities import (
    copy_graph,
    unify_edge_types_directed_undirected,
    general_graph_to_sem_model,
    get_nodes_from_node_names,
    add_edge_coefficients_from_sem_fit,
    add_edge_probabilities_to_graph, add_psyr_structural_equation_to_edge_coefficients,
)
from causal_pipe.utilities.plot_utilities import plot_correlation_graph
from causal_pipe.pysr.pysr_regression import symbolic_regression_causal_effect
from causal_pipe.pysr.cyclic_scm import CyclicSCMSimulator
from .pipe_config import (
    CausalPipeConfig,
    FASSkeletonMethod,
    BCSLSkeletonMethod,
    FCIOrientationMethod,
    HillClimbingOrientationMethod,
    VariableTypes,
    CausalEffectMethodNameEnum,
    CausalEffectMethod,
    PearsonCausalEffectMethod,
    SpearmanCausalEffectMethod,
    MICausalEffectMethod,
    KCICausalEffectMethod,
    SEMCausalEffectMethod,
    SEMClimbingCausalEffectMethod,
    PYSRCausalEffectMethod, HandlingMissingEnum,
)
from .pysr.pysr_hill_climber import search_best_graph_climber_pysr
from .utilities.utilities import dump_json_to, set_seed_python_and_r


class CausalPipe:
    """
    CausalPipe is a comprehensive pipeline for performing structural causal discovery and causal effect estimation.
    It handles data preprocessing, skeleton identification, edge orientation, and causal effect estimation.

    Features:
    - Data preprocessing: Handling missing values, encoding categorical variables, and feature selection.
    - Skeleton identification: Choose between FAS or BCSL methods.
    - Edge orientation: Use FCI or Hill Climbing algorithms.
    - Causal effect estimation: Utilize methods like Partial Linear Correlation and Partial Nonlinear Correlation.
    - Visualization: Generate plots for correlation graphs, skeletons, and final DAGs.
    """

    def __init__(self, config: CausalPipeConfig):
        """
        Initialize the CausalPipe.

        Parameters:
        - config (CausalPipeConfig): Comprehensive configuration for the toolkit.
        """
        # Initialize error logging
        self.errors: List[str] = []

        # Variable types
        if isinstance(config.variable_types, dict):
            config.variable_types = VariableTypes(**config.variable_types)
        self.variable_types = config.variable_types
        self.filtered_variables = []

        # Method configurations
        self.preprocessing_params = config.preprocessing_params
        self.skeleton_method = config.skeleton_method
        self.orientation_method = config.orientation_method
        self.causal_effect_methods = [
            self._convert_causal_effect_method(m)
            for m in (config.causal_effect_methods or [])
        ]

        # General settings
        self.show_plots = config.show_plots
        self.study_name = config.study_name
        self.root_output_folder = config.output_path
        self.output_path = os.path.join(self.root_output_folder, self.study_name)
        self.verbose = config.verbose
        self.seed = config.seed

        # Set random seed
        set_seed_python_and_r(self.seed)

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Placeholders for intermediate results
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.undirected_graph: Optional[GeneralGraph] = None
        self.sepsets: Dict[Tuple[int, int], Set[int]] = {}
        self.directed_graph: Optional[GeneralGraph] = None
        self.causal_effects: Dict[str, Any] = {}

    def _convert_causal_effect_method(self, method: CausalEffectMethod) -> CausalEffectMethod:
        """Convert a generic ``CausalEffectMethod`` into a specific subclass."""

        if method.__class__ is CausalEffectMethod:
            params = dict(method.params or {})
            common_args = {"directed": method.directed}
            if method.name == CausalEffectMethodNameEnum.PEARSON:
                return PearsonCausalEffectMethod(**common_args)
            if method.name == CausalEffectMethodNameEnum.SPEARMAN:
                return SpearmanCausalEffectMethod(**common_args)
            if method.name == CausalEffectMethodNameEnum.MI:
                return MICausalEffectMethod(**common_args)
            if method.name == CausalEffectMethodNameEnum.KCI:
                return KCICausalEffectMethod(**common_args)
            if method.name == CausalEffectMethodNameEnum.SEM:
                return SEMCausalEffectMethod(**common_args, **params)
            if method.name == CausalEffectMethodNameEnum.SEM_CLIMBING:
                return SEMClimbingCausalEffectMethod(**common_args, **params)
            if method.name == CausalEffectMethodNameEnum.PYSR:
                known = {
                    k: params.pop(k)
                    for k in [
                        "noise_kind",
                        "alpha",
                        "tol",
                        "max_iter",
                        "restarts",
                        "standardized_init",
                        "hc_orient_undirected_edges",
                    ]
                    if k in params
                }
                return PYSRCausalEffectMethod(
                    **common_args, pysr_params=params, **known
                )
        return method

    def _setup_logging(self):
        """
        Set up the logging configuration.
        """
        self.logger = logging.getLogger(self.study_name)
        self.logger.setLevel(logging.ERROR)

        # Create handlers
        log_file = os.path.join(self.output_path, "error.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)

        # Create formatter and add it to handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _log_error(self, method_name: str, exception: Exception):
        """
        Log an error message with traceback.

        Parameters:
        - method_name (str): Name of the method where the error occurred.
        - exception (Exception): The exception that was raised.
        """
        error_trace = traceback.format_exc()
        error_msg = (
            f"Error in {method_name}: {str(exception)}\nTraceback:\n{error_trace}"
        )
        self.errors.append(error_msg)
        self.logger.error(error_msg)
        if self.verbose:
            print(error_msg)

    def show_errors(self):
        """
        Display all logged errors in a user-friendly format.
        """
        if not self.errors:
            print("No errors encountered.")
            return

        print("\n=== Pipeline Errors ===")
        for idx, error in enumerate(self.errors, 1):
            print(f"\nError {idx}:\n{error}")
        print("=======================\n")

    def has_errors(self) -> bool:
        """
        Check if any errors have been logged.

        Returns:
        - bool: True if there are errors, False otherwise.
        """
        return len(self.errors) > 0

    def preprocess_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess the input DataFrame based on the specified parameters.

        Steps:
        - Handle missing values and impute using 'MICE'.
        - Prepare the data for mixed model, including categorical and float columns.
        - Prepare the data for causal discovery.

        Parameters:
        - df (pd.DataFrame): Raw input data.

        Returns:
        - Optional[pd.DataFrame]: Preprocessed data ready for causal discovery, or None if an error occurred.
        """
        method_name = "preprocess_data"
        try:
            print("Starting data preprocessing...")

            # Define variable types
            continuous_vars = self.variable_types.continuous
            ordinal_vars = self.variable_types.ordinal
            nominal_vars = self.variable_types.nominal
            all_vars = continuous_vars + ordinal_vars + nominal_vars

            if not all_vars:
                raise ValueError(
                    "No variables specified in variable_types. Please define at least one variable."
                )

            df_prepared = df.copy()

            if not self.preprocessing_params.no_preprocessing:
                # Prepare data for mixed model
                print("Preparing data for mixed model...")
                df_prepared = ensure_data_types(
                    df_prepared,
                    categorical_cols=nominal_vars + ordinal_vars,
                    float_cols=continuous_vars,
                    cat_to_codes=self.preprocessing_params.cat_to_codes,
                    standardize=self.preprocessing_params.standardize,
                )
                df_prepared = df_prepared.reset_index(drop=True)

                # Handle missing values and imputation using 'MICE'
                n_missing = df_prepared.isnull().sum().sum()
                if n_missing > 0:
                    print(f"Found {n_missing} missing values in the dataset.")
                    if self.preprocessing_params.handling_missing == HandlingMissingEnum.DROP:
                        print("Dropping rows with missing values...")
                        df_prepared = df_prepared.dropna()
                    elif self.preprocessing_params.handling_missing == HandlingMissingEnum.IMPUTE:
                        print(
                            f"Performing data imputation using {self.preprocessing_params.imputation_method}..."
                        )
                        if self.preprocessing_params.imputation_method == "mice":
                            warnings.warn(
                                "MICE imputation not implemented fully, no pooling across multiple imputations yet.",
                                UserWarning,
                            )
                        mice_dfs = perform_multiple_imputation(
                            df_prepared,
                            impute_cols=continuous_vars + nominal_vars + ordinal_vars,
                            full_obs_cols=self.preprocessing_params.full_obs_cols,
                            categorical_cols=nominal_vars + ordinal_vars,
                            method=self.preprocessing_params.imputation_method,
                            r_mice=self.preprocessing_params.use_r_mice,
                        )
                        # Use the first imputed dataset
                        # TODO: pooling results across multiple imputations
                        df_prepared = mice_dfs[0]
                    else:
                        raise ValueError(
                            f"Unsupported missing value handling method: {self.preprocessing_params.handling_missing}"
                        )

                # Check for empty features
                empty_features = df_prepared.columns[df_prepared.isnull().all()]
                if len(empty_features) > 0:
                    raise ValueError(
                        f"Empty features found after imputation: {empty_features}"
                    )

                # Prepare data for causal discovery
                print("Preparing data for causal discovery...")
                initial_columns = set(list(df_prepared.columns))
                df_prepared = prepare_data_for_causal_discovery(
                    df_prepared,
                    handle_missing="error",
                    encode_categorical=self.preprocessing_params.cat_to_codes,
                    scale_data=self.preprocessing_params.standardize,
                    keep_only_correlated_with=self.preprocessing_params.keep_only_correlated_with,
                    filter_method=self.preprocessing_params.filter_method,
                    filter_threshold=self.preprocessing_params.filter_threshold,
                )
                self.filtered_variables = list(
                    initial_columns - set(list(df_prepared.columns))
                )
                if self.filtered_variables:
                    print(
                        f"Filtered out variables: {self.filtered_variables} due to low "
                        f"correlation with {self.preprocessing_params.keep_only_correlated_with} "
                        f"- using {self.preprocessing_params.filter_method} filter."
                    )

            # Perform data validity checks
            test_results = perform_data_validity_checks(df_prepared)
            if self.output_path:
                with open(
                    os.path.join(self.output_path, "data_validity_checks.txt"), "w"
                ) as f:
                    f.write(f"{test_results}")

            self.preprocessed_data = df_prepared
            print("Data preprocessing completed.")
            return self.preprocessed_data

        except Exception as e:
            self._log_error(method_name, e)
            return None

    def identify_skeleton(
        self, df: Optional[pd.DataFrame] = None, show_plots: Optional[bool] = None
    ) -> Optional[Tuple[GeneralGraph, Dict[Tuple[int, int], Set[int]]]]:
        """
        Identify the global skeleton of the causal graph using the specified method.

        Parameters:
        - df (Optional[pd.DataFrame]): Raw input data. If None, uses preprocessed data.
        - show_plots (Optional[bool]): Whether to display plots. Overrides the default setting.

        Returns:
        - Optional[Tuple[GeneralGraph, Dict[Tuple[int, int], Set[int]]]]: The undirected graph and sepsets, or None if an error occurred.
        """
        method_name = "identify_skeleton"
        try:
            if df is not None:
                print("Preprocessing data...")
                self.preprocess_data(df)
            else:
                if self.preprocessed_data is None:
                    raise ValueError(
                        "Data must be preprocessed before identifying skeleton."
                    )

            if show_plots is None:
                show_plots = self.show_plots

            print(
                f"Identifying global skeleton using {self.skeleton_method.name} method..."
            )
            df = self.preprocessed_data

            if isinstance(self.skeleton_method, BCSLSkeletonMethod):
                bcsl = BCSL(
                    data=df,
                    num_bootstrap_samples=self.skeleton_method.bootstrap_resamples,
                    conditional_independence_method=self.skeleton_method.conditional_independence_method,
                    multiple_comparison_correction=self.skeleton_method.multiple_comparison_correction,
                    bootstrap_all_edges=self.skeleton_method.bootstrap_all_edges,
                    use_aee_alpha=self.skeleton_method.use_aee_alpha,
                    max_k=self.skeleton_method.max_k,
                    verbose=self.verbose,
                )
                self.undirected_graph = bcsl.combine_local_to_global_skeleton(
                    bootstrap_all_edges=True
                )
                self.sepsets = bcsl.sepsets

                print("Global skeleton (resolved):", bcsl.global_skeleton)
                visualize_graph(
                    self.undirected_graph,
                    title="BCSL Global Skeleton",
                    show=show_plots,
                    output_path=os.path.join(
                        self.output_path, "BCSL_Global_Skeleton.png"
                    ),
                )
            elif isinstance(self.skeleton_method, FASSkeletonMethod):
                if self.skeleton_method.conditional_independence_method == "gsq":
                    raise NotImplementedError(
                        "GSQ method is not yet supported for skeleton identification."
                    )
                # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
                nodes = get_nodes_from_node_names(node_names=list(df.columns))
                cit_method = CIT(
                    data=df.values,
                    method=self.skeleton_method.conditional_independence_method,
                )
                graph, sepsets, test_results = fas(
                    data=df.values,
                    nodes=nodes,
                    independence_test_method=cit_method,
                    alpha=self.skeleton_method.alpha,
                    knowledge=self.skeleton_method.knowledge,
                    depth=self.skeleton_method.depth,
                    show_progress=self.verbose,
                )
                self.undirected_graph = graph
                self.sepsets = sepsets

                print("Global skeleton (FAS):", graph)
                visualize_graph(
                    self.undirected_graph,
                    title="FAS Global Skeleton",
                    # labels=dict(zip(range(len(df.columns)), df.columns)),
                    show=show_plots,
                    output_path=os.path.join(
                        self.output_path, "FAS_Global_Skeleton.png"
                    ),
                )
                if self.skeleton_method.bootstrap_resamples > 0:
                    fas_kwargs = dict(
                        alpha=self.skeleton_method.alpha,
                        depth=self.skeleton_method.depth,
                        knowledge=self.skeleton_method.knowledge,
                        conditional_independence_method=self.skeleton_method.conditional_independence_method,
                    )
                    (
                        self.fas_edge_probabilities,
                        self.best_graph_with_fas_bootstrap,
                    ) = bootstrap_fas_edge_stability(
                        df,
                        resamples=self.skeleton_method.bootstrap_resamples,
                        random_state=self.skeleton_method.bootstrap_random_state,
                        fas_kwargs=fas_kwargs,
                        output_dir=os.path.join(self.output_path, "fas_bootstrap"),
                        n_jobs=self.skeleton_method.n_jobs,
                    )
                    oriented_probs = {
                        k: {"TAIL-TAIL": v} for k, v in self.fas_edge_probabilities.items()
                    }
                    prob_graph, edges_with_probabilities = add_edge_probabilities_to_graph(
                        self.undirected_graph, oriented_probs
                    )
                    visualize_graph(
                        prob_graph,
                        edges=edges_with_probabilities,
                        title="FAS Graph with Bootstrap Probabilities",
                        show=show_plots,
                        output_path=os.path.join(
                            self.output_path,
                            "fas_bootstrap",
                            "full_graph.png",
                        ),
                    )
                    if self.skeleton_method.bootstrap_edge_threshold is not None:
                        threshold = self.skeleton_method.bootstrap_edge_threshold
                        node_names = list(df.columns)
                        filtered_edges = []
                        for edge in self.undirected_graph.get_graph_edges():
                            a = edge.get_node1().get_name()
                            b = edge.get_node2().get_name()
                            if (
                                self.fas_edge_probabilities.get((a, b), 0.0)
                                >= threshold
                            ):
                                filtered_edges.append((a, b, "TAIL", "TAIL"))
                        self.undirected_graph = make_graph(node_names, filtered_edges)
                        prob_graph, edges_with_probabilities = add_edge_probabilities_to_graph(
                            self.undirected_graph, oriented_probs
                        )
                        visualize_graph(
                            prob_graph,
                            edges=edges_with_probabilities,
                            title=f"FAS Graph with Bootstrap Probabilities (threshold={threshold})",
                            show=show_plots,
                            output_path=os.path.join(
                                self.output_path,
                                "fas_bootstrap",
                                "full_graph_thresholded.png",
                            ),
                        )
            else:
                raise ValueError(
                    f"Unsupported skeleton method: {self.skeleton_method.name}"
                )

            print("Skeleton identification completed.")
            return self.undirected_graph, self.sepsets

        except Exception as e:
            self._log_error(method_name, e)
            return None

    def orient_edges(
        self, df: Optional[pd.DataFrame] = None, show_plot: bool = False
    ) -> Optional[GeneralGraph]:
        """
        Orient the edges of the skeleton using the specified orientation method.

        Parameters:
        - df (Optional[pd.DataFrame]): Raw input data. If None, uses preprocessed data.
        - show_plot (bool): Whether to display the resulting graph.

        Returns:
        - Optional[GeneralGraph]: The directed graph, or None if an error occurred.
        """
        method_name = "orient_edges"
        try:
            if df is not None:
                self.preprocess_data(df)
                self.identify_skeleton()
            else:
                if self.undirected_graph is None:
                    if self.preprocessed_data is None:
                        raise ValueError(
                            "Data must be preprocessed before orienting edges."
                        )
                    self.identify_skeleton()

            print(f"Orienting edges using {self.orientation_method.name} method...")
            df = self.preprocessed_data

            if isinstance(self.orientation_method, FCIOrientationMethod):
                graph_fci, edges_fci = fci_orient_edges_from_graph_node_sepsets(
                    data=df.values,
                    graph=copy_graph(self.undirected_graph),
                    nodes=self.undirected_graph.nodes,
                    sepsets=self.sepsets,
                    background_knowledge=self.orientation_method.background_knowledge,
                    independence_test_method=self.orientation_method.conditional_independence_method,
                    alpha=self.orientation_method.alpha,
                    max_path_length=self.orientation_method.max_path_length,
                    verbose=self.verbose,
                )
                visualize_graph(
                    graph_fci,
                    title="Causal Learn FCI Result",
                    show=show_plot,
                    output_path=os.path.join(self.output_path, "FCI_Result.png"),
                )
                self.directed_graph = graph_fci
            elif isinstance(self.orientation_method, HillClimbingOrientationMethod):
                bcsl = BCSL(
                    data=df,
                    verbose=self.verbose,
                )
                self.directed_graph = bcsl.orient_edges_hill_climbing(
                    undirected_graph=copy_graph(self.undirected_graph)
                )
                visualize_graph(
                    self.directed_graph,
                    title="Hill Climbing Oriented Graph",
                    show=show_plot,
                    output_path=os.path.join(
                        self.output_path, "Hill_Climbing_Result.png"
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported orientation method: {self.orientation_method.name}"
                )

            print("Edge orientation completed.")
            return self.directed_graph

        except Exception as e:
            self._log_error(method_name, e)
            return None

    def estimate_causal_effects(
        self, df: Optional[pd.DataFrame] = None, show_plot: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate causal effects using the specified methods.

        Parameters:
        - df (Optional[pd.DataFrame]): Raw input data. If None, uses preprocessed data.
        - show_plot (bool): Whether to display plots.

        Returns:
        - Optional[Dict[str, Any]]: The estimated causal effects, or None if an error occurred.
        """
        method_name = "estimate_causal_effects"
        try:
            if self.causal_effect_methods is None:
                print("No causal effect estimation methods specified.")
                self.causal_effects = None
                return None

            if self.directed_graph is None:
                raise ValueError(
                    "Edges must be oriented before estimating causal effects."
                )

            print("Estimating causal effects...")
            if df is not None:
                self.preprocess_data(df)
                self.identify_skeleton()
                self.orient_edges()
            else:
                if self.preprocessed_data is None:
                    raise ValueError(
                        "Data must be preprocessed before estimating causal effects."
                    )

            df = self.preprocessed_data

            for method in self.causal_effect_methods:
                if method.name in ["pearson", "spearman", "mi", "kci"]:
                    # Partial Correlation / MI / KCI
                    graph = (
                        self.directed_graph
                        if method.directed
                        else self.undirected_graph
                    )
                    self.causal_effects[method.name] = compute_partial_correlations(
                        df, method=method.name, known_graph=graph
                    )
                    out_dir = os.path.join(
                        self.output_path, "causal_effect", method.name
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    plot_correlation_graph(
                        self.causal_effects[method.name],
                        labels=df.columns,
                        threshold=0.001,
                        layout="hierarchical",
                        auto_order=True,
                        node_size=2500,
                        node_color="lightblue",
                        font_size=12,
                        edge_cmap="bwr",
                        edge_vmin=-1,
                        edge_vmax=1,
                        min_edge_width=1,
                        max_edge_width=5,
                        title=f"{method.name.capitalize()} Partial Correlation Graph",
                        output_path=os.path.join(out_dir, f"{method.name}_result.png"),
                        show=show_plot,
                    )
                    dump_json_to(
                        data=self.causal_effects[method.name],
                        path=os.path.join(out_dir, f"{method.name}_results.json"),
                    )
                elif isinstance(method, SEMCausalEffectMethod):
                    # Structural Equation Modeling
                    directed_graph = unify_edge_types_directed_undirected(
                        self.directed_graph
                    )
                    model_str, exogenous_vars = general_graph_to_sem_model(
                        directed_graph
                    )

                    ordered = self.get_ordered_variable_names()
                    default_estimator = "ML"
                    if ordered:
                        default_estimator = "WLSMV"

                    estimator = method.estimator or default_estimator
                    self.causal_effects[method.name] = fit_sem_lavaan(
                        df,
                        model_str,
                        var_names=None,
                        estimator=estimator,
                        ordered=ordered,
                        exogenous_vars_model_1=exogenous_vars,
                    )
                    coef_graph, edges_with_coefficients = (
                        add_edge_coefficients_from_sem_fit(
                            directed_graph,
                            model_output=self.causal_effects[method.name],
                        )
                    )
                    out_sem_dir = os.path.join(self.output_path, "causal_effect", "sem")
                    os.makedirs(out_sem_dir, exist_ok=True)
                    visualize_graph(
                        coef_graph,
                        edges=edges_with_coefficients,
                        title="SEM Result",
                        show=show_plot,
                        output_path=os.path.join(
                            out_sem_dir, "sem_result_with_coefficients.png"
                        ),
                    )
                    visualize_graph(
                        directed_graph,
                        title="SEM Result",
                        show=show_plot,
                        output_path=os.path.join(
                            out_sem_dir, "sem_result_without_coefficients.png"
                        ),
                    )
                    dump_json_to(
                        data=self.causal_effects[method.name],
                        path=os.path.join(out_sem_dir, "sem_results.json"),
                    )
                elif isinstance(method, SEMClimbingCausalEffectMethod):
                    # Structural Equation Modeling with Hill Climbing
                    ordered = self.get_ordered_variable_names()
                    default_estimator = "ML"
                    if ordered:
                        # default_estimator = "WLSMV"
                        default_estimator = "MLR"
                        warnings.warn(
                            "Ordered variables detected but not supported by SEM Climber. Using MLR estimator instead."
                        )
                    est = method.estimator or default_estimator
                    respect_pag = method.respect_pag
                    finalize_with_resid_covariances = (
                        method.finalize_with_resid_covariances
                    )
                    best_graph, sem_results = search_best_graph_climber(
                        df,
                        initial_graph=self.directed_graph,
                        node_names=list(df.columns),
                        max_iter=method.max_iter,
                        estimator=est,
                        ordered=ordered,
                        finalize_with_resid_covariances=finalize_with_resid_covariances,
                        respect_pag=respect_pag,
                    )
                    self.causal_effects[method.name] = {
                        "best_graph": best_graph,
                        "summary": sem_results,
                        "resid_cov_aug": sem_results.get("resid_cov_aug"),
                    }
                    if method.chain_orientation:
                        print("[Causal Pipe] Chain Orientation - Saving best graph from SEM Climber as directed graph "
                              "for future CE estimation.")
                        self.directed_graph = best_graph

                    print("Saving results to output directory.")
                    out_sem_dir = os.path.join(
                        self.output_path, "causal_effect", "sem_climber"
                    )
                    os.makedirs(out_sem_dir, exist_ok=True)
                    visualize_graph(
                        best_graph,
                        title="Best Graph Climber Result",
                        show=show_plot,
                        output_path=os.path.join(out_sem_dir, "best_graph.png"),
                    )
                    coef_graph, edges_with_coefficients = (
                        add_edge_coefficients_from_sem_fit(
                            best_graph,
                            model_output=self.causal_effects[method.name]["summary"],
                        )
                    )
                    visualize_graph(
                        coef_graph,
                        edges=edges_with_coefficients,
                        title="Best Graph Climber Result With Coefficients",
                        show=show_plot,
                        output_path=os.path.join(
                            out_sem_dir, "best_graph_with_coefficients.png"
                        ),
                    ) 
                    if sem_results is None:
                        sem_results = {"fit_summary": "Failure"}
                    # Keys to export
                    keys_to_extract = [
                        "estimator",
                        "model_1_string",
                        "fit_summary",
                        "fit_measures",
                        "unstandardized_parameter_estimates",
                        "standardized_parameter_estimates",
                        "measurement_model",
                        "structural_model",
                        "residual_covariances",
                        "factor_scores",
                        "r2",
                        "log_likelihood",
                        "log_likelihoods",
                        "npar",
                        "n_samples",
                        "comparison_results",
                        "is_better_model",
                        "model_2_string",
                    ]
                    sem_results_to_dump = {
                        k: v for k, v in sem_results.items() if k in keys_to_extract
                    }
                    dump_json_to(
                        data=sem_results_to_dump,
                        path=os.path.join(out_sem_dir, "sem_climber_results.json"),
                    )
                    with open(os.path.join(out_sem_dir, "fit_summary.txt"), "w") as f:
                        f.write(f"{sem_results.get('fit_summary')}")
                elif isinstance(method, PYSRCausalEffectMethod):
                    # Output directory for PySR results
                    out_dir = os.path.join(
                        self.output_path, "causal_effect", method.name
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    # Symbolic regression using PySR
                    graph = (
                        self.directed_graph
                        if method.directed
                        else self.undirected_graph
                    )
                    pysr_params = dict(method.pysr_params)
                    pysr_params["output_directory"] = os.path.join(out_dir, "pysr_output")
                    if method.hc_orient_undirected_edges:
                        # Run Hill Climbing
                        best_graph, best_score = search_best_graph_climber_pysr(
                            df,
                            initial_graph=graph,
                            pysr_params=pysr_params,
                            estimator=method.estimator,
                            max_iter=method.hc_max_iter,
                            respect_pag=method.respect_pag,
                            out_dir=out_dir
                        )
                        self.causal_effects[method.name] = {
                            "final_graph": best_graph,
                            "best_score": best_score,
                            "structural_equations": best_score.get("structural_equations"),
                            "fit_measures": best_score.get("fit_measures"),
                        }
                        if method.chain_orientation:
                            print("[Causal Pipe] Chain Orientation - Saving best graph from Hill Climbing as directed "
                                  "graph for future CE estimation.")
                            self.directed_graph = best_graph
                    else:
                        pysr_fitter_output = symbolic_regression_causal_effect(
                            df,
                            graph,
                            pysr_params=pysr_params,
                        )
                        self.causal_effects[method.name] = {
                            "final_graph": pysr_fitter_output.final_graph,
                            "structural_equations": pysr_fitter_output.structural_equations,
                        }
                        simulator = CyclicSCMSimulator(
                            structural_equations=pysr_fitter_output.structural_equations,
                            undirected_graph=self.undirected_graph,
                            df_columns=list(df.columns),
                            seed=self.seed,
                            out_dir=out_dir
                        )
                        residuals, Omega, resid_rows = simulator.estimate_noise(
                            df
                        )
                        sim_data, solver_stats = simulator.simulate(
                            df,
                            Omega=Omega,
                            resid_rows=resid_rows,
                            out_dir=out_dir,
                            noise_kind=method.noise_kind,
                            alpha=method.alpha,
                            tol=method.tol,
                            max_iter=method.max_iter,
                            restarts=method.restarts,
                            standardized_init=method.standardized_init,
                        )
                        self.causal_effects[method.name][
                            "fit_measures"
                        ] = simulator.compute_fit_measures(
                            df, sim_data, residuals, solver_stats
                        )
                    dump_json_to(
                        data=self.causal_effects[method.name],
                        path=os.path.join(out_dir, f"{method.name}_results.json"),
                    )
                    coef_graph, edges, structural_equations = (
                        add_psyr_structural_equation_to_edge_coefficients(
                            final_graph=self.causal_effects[method.name]["final_graph"],
                            structural_equations=self.causal_effects[method.name]["structural_equations"],
                        )
                    )
                    graph_name = "pysr_result_with_equations.png"
                    if method.hc_orient_undirected_edges:
                        graph_name = "pysr_result_hc_best_graph_with_equations.png"

                    visualize_graph(
                        coef_graph,
                        edges=edges,
                        structural_equations=structural_equations,
                        title="PySR SCM Result",
                        show=show_plot,
                        output_path=os.path.join(
                            out_dir, graph_name
                        ),
                    )
                    dump_json_to(
                        self.causal_effects[method.name]["fit_measures"], os.path.join(out_dir, "fit_measures.json")
                    )
                else:
                    raise ValueError(
                        f"Unsupported causal effect estimation method: {method.name}"
                    )

            print("Causal effect estimation completed.")
            return self.causal_effects

        except Exception as e:
            self._log_error(method_name, e)
            return None

    def get_ordered_variable_names(self) -> List[str]:
        """
        Get the ordinal and nominal variable names.

        Returns:
        - List[str]: Variable names.
        """
        ordinal_vars = self.variable_types.ordinal
        nominal_vars = self.variable_types.nominal
        initial_vars = ordinal_vars + nominal_vars
        if self.filtered_variables and any(
            var in self.filtered_variables for var in initial_vars
        ):
            print(
                "-- Some ordinal/nominal variables were filtered out: ",
                self.filtered_variables,
            )
            return [var for var in initial_vars if var not in self.filtered_variables]
        return initial_vars

    def run_pipeline(self, df: pd.DataFrame):
        """
        Execute the full causal discovery pipeline: preprocessing, skeleton identification, edge orientation, and causal effect estimation.

        Parameters:
        - df (pd.DataFrame): Raw input data.
        """
        method_name = "run_pipeline"
        try:
            print("Starting the full causal discovery pipeline...")
            self.preprocess_data(df)
            if self.has_errors():
                raise RuntimeError("Preprocessing failed. Aborting pipeline.")

            self.identify_skeleton()
            if self.has_errors():
                raise RuntimeError("Skeleton identification failed. Aborting pipeline.")

            self.orient_edges()
            if self.has_errors():
                raise RuntimeError("Edge orientation failed. Aborting pipeline.")

            self.estimate_causal_effects()
            if self.has_errors():
                raise RuntimeError(
                    "Causal effect estimation failed. Aborting pipeline."
                )

            print("Causal discovery pipeline completed successfully.")

        except Exception as e:
            self._log_error(method_name, e)
            print(
                "Causal discovery pipeline terminated due to errors. Use 'show_errors()' to view them."
            )
