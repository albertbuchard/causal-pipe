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
from causal_pipe.sem.sem import fit_sem_lavaan, search_best_graph_climber
from causal_pipe.utilities.graph_utilities import (
    copy_graph,
    unify_edge_types_directed_undirected,
    general_graph_to_sem_model,
    get_nodes_from_node_names,
    add_edge_coefficients_from_sem_fit,
)
from causal_pipe.utilities.plot_utilities import plot_correlation_graph
from .pipe_config import (
    CausalPipeConfig,
    FASSkeletonMethod,
    BCSLSkeletonMethod,
    FCIOrientationMethod,
    HillClimbingOrientationMethod,
    VariableTypes,
)
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
        self.causal_effect_methods = config.causal_effect_methods

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
                    if self.preprocessing_params.handling_missing == "drop":
                        print("Dropping rows with missing values...")
                        df_prepared = df_prepared.dropna()
                    elif self.preprocessing_params.handling_missing == "impute":
                        print(
                            f"Performing data imputation using {self.preprocessing_params.imputation_method}..."
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
                    num_bootstrap_samples=self.skeleton_method.num_bootstrap_samples,
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
                    labels=dict(zip(range(len(df.columns)), df.columns)),
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
                    labels=dict(zip(range(len(df.columns)), df.columns)),
                    show=show_plots,
                    output_path=os.path.join(
                        self.output_path, "FAS_Global_Skeleton.png"
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
                    labels=dict(zip(range(len(df.columns)), df.columns)),
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
                    labels=dict(zip(range(len(df.columns)), df.columns)),
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
                elif method.name == "sem":
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

                    self.causal_effects[method.name] = fit_sem_lavaan(
                        df,
                        model_str,
                        var_names=None,
                        estimator=method.params.get("estimator", default_estimator),
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
                        labels=dict(zip(range(len(df.columns)), df.columns)),
                        show=show_plot,
                        output_path=os.path.join(
                            out_sem_dir, "sem_result_with_coefficients.png"
                        ),
                    )
                    visualize_graph(
                        directed_graph,
                        title="SEM Result",
                        labels=dict(zip(range(len(df.columns)), df.columns)),
                        show=show_plot,
                        output_path=os.path.join(
                            out_sem_dir, "sem_result_without_coefficients.png"
                        ),
                    )
                    dump_json_to(
                        data=self.causal_effects[method.name],
                        path=os.path.join(out_sem_dir, "sem_results.json"),
                    )
                elif method.name == "sem-climbing":
                    # Structural Equation Modeling with Hill Climbing
                    ordered = self.get_ordered_variable_names()
                    default_estimator = "ML"
                    if ordered:
                        # default_estimator = "WLSMV"
                        default_estimator = "MLR"
                        warnings.warn(
                            "Ordered variables detected but not supported by SEM Climber. Using MLR estimator instead."
                        )
                    est = method.params.get("estimator", default_estimator)
                    respect_pag = bool(method.params.get("respect_pag", True))
                    best_graph, sem_results = search_best_graph_climber(
                        df,
                        initial_graph=self.directed_graph,
                        node_names=list(df.columns),
                        max_iter=method.params.get("max_iter", 100),
                        estimator=est,
                        ordered=ordered,
                        finalize_with_resid_covariances=method.params.get(
                            "finalize_with_resid_covariances", False
                        ),
                        mi_cutoff=method.params.get("mi_cutoff", 10.0),
                        sepc_cutoff=method.params.get("sepc_cutoff", 0.10),
                        max_add=method.params.get("max_add", 5),
                        delta_stop=method.params.get("delta_stop", 0.003),
                        whitelist_pairs=method.params.get("whitelist_pairs"),
                        forbid_pairs=method.params.get("forbid_pairs"),
                        same_occasion_regex=method.params.get("same_occasion_regex"),
                        respect_pag=respect_pag,
                    )
                    self.causal_effects[method.name] = {
                        "best_graph": best_graph,
                        "summary": sem_results,
                        "resid_cov_aug": sem_results.get("resid_cov_aug"),
                    }

                    print("Saving results to output directory.")
                    out_sem_dir = os.path.join(
                        self.output_path, "causal_effect", "sem_climber"
                    )
                    os.makedirs(out_sem_dir, exist_ok=True)
                    visualize_graph(
                        best_graph,
                        title="Best Graph Climber Result",
                        labels=dict(zip(range(len(df.columns)), df.columns)),
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
                        title="Best Graph Climber Result With Coefficients",
                        edges=edges_with_coefficients,
                        labels=dict(zip(range(len(df.columns)), df.columns)),
                        show=show_plot,
                        output_path=os.path.join(
                            out_sem_dir, "best_graph_with_coefficients.png"
                        ),
                    )
                    if sem_results is None:
                        sem_results = {"fit_summary": "Failure"}
                    dump_json_to(
                        data=sem_results,
                        path=os.path.join(out_sem_dir, "sem_climber_results.json"),
                    )
                    with open(os.path.join(out_sem_dir, "fit_summary.txt"), "w") as f:
                        f.write(f"{sem_results.get('fit_summary')}")

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
