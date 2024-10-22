# API Reference

Welcome to the **CausalPipe** API Reference. This section provides detailed information about the classes, methods, and functions available in the **CausalPipe** package. Whether you're integrating **CausalPipe** into your workflow or extending its functionality, this guide will help you navigate its components effectively.

## Table of Contents

- [Classes](#classes)
  - [CausalPipe](#causalpipe)
  - [CausalPipeConfig](#causalpipeconfig)
    - [VariableTypes](#variabletypes)
    - [DataPreprocessingParams](#datapreprocessingparams)
    - [SkeletonMethod](#skeletonmethod)
    - [BCSLSkeletonMethod](#bcslmethod)
    - [FASSkeletonMethod](#fasskeletonmethod)
    - [OrientationMethod](#orientationmethod)
    - [FCIOrientationMethod](#fciorientationmethod)
    - [HillClimbingOrientationMethod](#hillclimbingorientationmethod)
    - [CausalEffectMethod](#causaleffectmethod)
  - [SEMScore](#semscore)
- [Functions](#functions)
  - [fit_sem_lavaan](#fit_sem_lavaan)
  - [search_best_graph_climber](#search_best_graph_climber)

---

## Classes

### CausalPipe

**`CausalPipe`** is the core class of the **CausalPipe** package. It orchestrates the entire causal discovery pipeline, handling data preprocessing, skeleton identification, edge orientation, and causal effect estimation.

#### Initialization

```python
CausalPipe(config: CausalPipeConfig)
```

- **Parameters:**
    - `config` (`CausalPipeConfig`): A comprehensive configuration object that defines variable types, preprocessing parameters, skeleton identification methods, edge orientation methods, and causal effect estimation methods.

#### Methods

#### **`preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]`**

  Preprocesses the input DataFrame based on the specified parameters in the configuration. This includes handling missing values, encoding categorical variables, standardizing features, and performing feature selection.

  - **Parameters:**
      - `df` (`pd.DataFrame`): Raw input data.

  - **Returns:**
      - `Optional[pd.DataFrame]`: Preprocessed data ready for causal discovery, or `None` if an error occurred.

#### **`identify_skeleton(df: Optional[pd.DataFrame] = None, show_plots: Optional[bool] = None) -> Optional[Tuple[GeneralGraph, Dict[Tuple[int, int], Set[int]]]]`**

  Identifies the global skeleton of the causal graph using the specified method (e.g., FAS or BCSL).

  - **Parameters:**
      - `df` (`Optional[pd.DataFrame]`): Raw input data. If `None`, uses preprocessed data.
      - `show_plots` (`Optional[bool]`): Whether to display plots. Overrides the default setting.

  - **Returns:**
      - `Optional[Tuple[GeneralGraph, Dict[Tuple[int, int], Set[int]]]]`: The undirected graph and sepsets, or `None` if an error occurred.

#### **`orient_edges(df: Optional[pd.DataFrame] = None, show_plot: bool = False) -> Optional[GeneralGraph]`**

  Orients the edges of the skeleton using the specified orientation method (e.g., FCI or Hill Climbing).

  - **Parameters:**
      - `df` (`Optional[pd.DataFrame]`): Raw input data. If `None`, uses preprocessed data.
      - `show_plot` (`bool`): Whether to display the resulting graph.

  - **Returns:**
      - `Optional[GeneralGraph]`: The directed graph, or `None` if an error occurred.

#### **`estimate_causal_effects(df: Optional[pd.DataFrame] = None, show_plot: bool = False) -> Optional[Dict[str, Any]]`**

  Estimates causal effects using the specified methods (e.g., Partial Correlation, SEM).

  - **Parameters:**
      - `df` (`Optional[pd.DataFrame]`): Raw input data. If `None`, uses preprocessed data.
      - `show_plot` (`bool`): Whether to display plots.

  - **Returns:**
      - `Optional[Dict[str, Any]]`: The estimated causal effects, or `None` if an error occurred.

#### **`run_pipeline(df: pd.DataFrame)`**

  Executes the full causal discovery pipeline: preprocessing, skeleton identification, edge orientation, and causal effect estimation.

  - **Parameters:**
    - `df` (`pd.DataFrame`): Raw input data.

#### **`show_errors()`**

  Displays all logged errors in a user-friendly format.

#### **`has_errors() -> bool`**

  Checks if any errors have been logged.

  - **Returns:**
    - `bool`: `True` if there are errors, `False` otherwise.

#### **`get_ordered_variable_names() -> List[str]`**

  Retrieves the names of ordinal and nominal variables.

  - **Returns:**
    - `List[str]`: List of ordered variable names.

---

### CausalPipeConfig

**`CausalPipeConfig`** is a dataclass that encapsulates all configuration parameters required to set up the **CausalPipe** pipeline.

#### Attributes

- `variable_types` (`VariableTypes`): Defines the types of variables (`continuous`, `ordinal`, `nominal`) in the dataset.
- `preprocessing_params` (`DataPreprocessingParams`): Parameters for data preprocessing.
- `skeleton_method` (`SkeletonMethod`): Method for skeleton identification (e.g., `FASSkeletonMethod`).
- `orientation_method` (`OrientationMethod`): Method for edge orientation (e.g., `FCIOrientationMethod`).
- `causal_effect_methods` (`List[CausalEffectMethod]`): List of methods for estimating causal effects (e.g., `CausalEffectMethod`).
- `study_name` (`str`): Name of the study, used in output file naming.
- `output_path` (`str`): Directory where results will be saved.
- `show_plots` (`bool`): Whether to show plots.
- `verbose` (`bool`): Enable verbose logging.

---

#### VariableTypes

**`VariableTypes`** defines the categorization of variables in your dataset.

- **Attributes:**
    - `continuous` (`List[str]`): List of continuous variable names.
    - `ordinal` (`List[str]`, default `[]`): List of ordinal variable names.
    - `nominal` (`List[str]`, default `[]`): List of nominal variable names.

#### DataPreprocessingParams

**`DataPreprocessingParams`** configures how data preprocessing is handled in the pipeline.

- **Attributes:**
    - `no_preprocessing` (`bool`, default `False`): Whether to skip preprocessing.
    - `handling_missing` (`str`, default `"impute"`): Method to handle missing values (`"impute"`, `"drop"`, etc.).
    - `cat_to_codes` (`bool`, default `True`): Whether to convert categorical variables to numeric codes.
    - `standardize` (`bool`, default `True`): Whether to standardize continuous variables.
    - `imputation_method` (`str`, default `"mice"`): Method for imputation (`"mice"`, etc.).
    - `use_r_mice` (`bool`, default `True`): Whether to use R's `mice` for imputation.
    - `full_obs_cols` (`Optional[List[str]]`, default `None`): Columns to keep as fully observed.
    - `keep_only_correlated_with` (`Optional[str]`, default `None`): Feature selection based on correlation with a target variable.
    - `filter_method` (`str`, default `"mi"`): Method for feature filtering (`"mi"`, `"pearson"`, etc.).
    - `filter_threshold` (`float`, default `0.1`): Threshold for feature filtering.
    - `kwargs` (`Optional[Dict[str, Any]]`, default `{}`): Additional keyword arguments.

#### SkeletonMethod

**`SkeletonMethod`** is a base dataclass for configuring skeleton identification methods.

- **Attributes:**
    - `name` (`str`): Name of the skeleton method.
    - `conditional_independence_method` (`str`, default `"fisherz"`): Method for conditional independence testing.
    - `alpha` (`float`, default `0.05`): Significance level for tests.
    - `params` (`Optional[Dict[str, Any]]`, default `{}`): Additional parameters.

#### BCSLSkeletonMethod

**`BCSLSkeletonMethod`** configures the Bootstrap-based Causal Structure Learning (BCSL) method for skeleton identification.

- **Inherits From:** `SkeletonMethod`

- **Attributes:**
    - `name` (`str`, default `"BCSL"`): Name of the skeleton method.
    - `num_bootstrap_samples` (`int`, default `100`): Number of bootstrap samples.
    - `multiple_comparison_correction` (`str`, default `"fdr"`): Method for multiple comparison correction (`"fdr"`, etc.).
    - `bootstrap_all_edges` (`bool`, default `True`): Whether to bootstrap all edges.
    - `use_aee_alpha` (`float`, default `0.05`): Alpha level for AEE.
    - `max_k` (`int`, default `3`): Maximum conditioning set size.

#### FASSkeletonMethod

**`FASSkeletonMethod`** configures the Fast Adjacency Search (FAS) method for skeleton identification.

- **Inherits From:** `SkeletonMethod`

- **Attributes:**
    - `name` (`str`, default `"FAS"`): Name of the skeleton method.
    - `depth` (`int`, default `3`): Depth parameter for FAS.
    - `knowledge` (`Optional[BackgroundKnowledge]`, default `None`): Background knowledge for FAS.

#### OrientationMethod

**`OrientationMethod`** is a base dataclass for configuring edge orientation methods.

- **Attributes:**
      - `name` (`str`, default `"FCI"`): Name of the orientation method.
      - `conditional_independence_method` (`str`, default `"fisherz"`): Method for conditional independence testing.

#### FCIOrientationMethod

**`FCIOrientationMethod`** configures the Fast Causal Inference (FCI) method for edge orientation.

- **Inherits From:** `OrientationMethod`

- **Attributes:**
    - `name` (`str`, default `"FCI"`): Name of the orientation method.
    - `background_knowledge` (`Optional[BackgroundKnowledge]`, default `None`): Background knowledge for FCI.
    - `alpha` (`float`, default `0.05`): Significance level for tests.
    - `max_path_length` (`int`, default `3`): Maximum path length for FCI.

#### HillClimbingOrientationMethod

**`HillClimbingOrientationMethod`** configures the Hill Climbing method for edge orientation.

- **Inherits From:** `OrientationMethod`

- **Attributes:**
  - `name` (`str`, default `"Hill Climbing"`): Name of the orientation method.
  - `max_k` (`int`, default `3`): Maximum conditioning set size.
  - `multiple_comparison_correction` (`str`, default `"fdr"`): Method for multiple comparison correction (`"fdr"`, etc.).

#### CausalEffectMethod

**`CausalEffectMethod`** configures the methods used for estimating causal effects.

- **Attributes:**
  - `name` (`str`, default `"pearson"`): Name of the causal effect estimation method.
    - **Options:**
      - `'pearson'`: Partial Pearson Correlation
      - `'spearman'`: Partial Spearman Correlation
      - `'mi'`: Conditional Mutual Information
      - `'kci'`: Kernel Conditional Independence
      - `'sem'`: Structural Equation Modeling
      - `'sem-climbing'`: Structural Equation Modeling with Hill Climbing search of the best graph.
  - `directed` (`bool`, default `True`): Indicates if the method uses a directed graph.
  - `params` (`Optional[Dict[str, Any]]`, default `{}`): Additional parameters for the method.

---

### SEMScore

**`SEMScore`** is a class used to score graph structures based on Structural Equation Modeling (SEM) fits. It integrates with **CausalPipe** to evaluate the quality of causal graphs.

#### Initialization

```python
SEMScore(
    data: pd.DataFrame,
    var_names: Optional[Dict[str, str]] = None,
    estimator: str = "MLR",
    return_metrics: bool = False,
    ordered: Optional[List[str]] = None,
)
```

- **Parameters:**
  - `data` (`pd.DataFrame`): The dataset used for SEM fitting.
  - `var_names` (`Optional[Dict[str, str]]`, default `None`): A dictionary mapping current factor names to meaningful names. Example: `{'Academic': 'Academic_Ability', 'Arts': 'Artistic_Skills'}`.
  - `estimator` (`str`, default `"MLR"`): The estimator to use for fitting the SEM model. Options include `"MLM"`, `"MLR"`, `"ULS"`, `"WLSMV"`, etc. `"bayesian"` is not yet implemented.
  - `return_metrics` (`bool`, default `False`): Whether to return additional fit metrics in the output.
  - `ordered` (`Optional[List[str]]`, default `None`): A list of variable names that are ordered (ordinal variables). Currently not implemented and will be ignored with a warning.

#### Methods

#### **`__call__(general_graph: GeneralGraph, compared_to_graph: Optional[GeneralGraph] = None) -> Dict[str, Any]`**

  Calculates the score of the given graph based on BIC from SEM fitting.

  - **Parameters:**
    - `general_graph` (`GeneralGraph`): The graph to score.
    - `compared_to_graph` (`Optional[GeneralGraph]`, default `None`): The graph to compare the given graph against.

  - **Returns:**
    - `Dict[str, Any]`: A dictionary containing the score and additional SEM fitting results.

####  **`exhaustive_results(general_graph: GeneralGraph, compared_to_graph: Optional[GeneralGraph] = None) -> Dict[str, Any]`**

  Fits an SEM model for the given graph and returns the results.

  - **Parameters:**
    - `general_graph` (`GeneralGraph`): The graph structure to fit.
    - `compared_to_graph` (`Optional[GeneralGraph]`, default `None`): The graph structure to compare the given graph against.

  - **Returns:**
    - `Dict[str, Any]`: A dictionary containing the SEM fitting results.

---

## Functions

### fit_sem_lavaan

**`fit_sem_lavaan`** fits a Structural Equation Model (SEM) using the specified model string and returns comprehensive results. It leverages R's `lavaan` package via `rpy2` for SEM fitting.

```python
fit_sem_lavaan(
    data: pd.DataFrame,
    model_1_string: str,
    var_names: Optional[Dict[str, str]] = None,
    estimator: str = "MLM",
    model_2_string: Optional[str] = None,
    ordered: Optional[List[str]] = None,
) -> Dict[str, Any]
```

- **Parameters:**
    - `data` (`pd.DataFrame`): The dataset including all variables needed for the SEM.
    - `model_1_string` (`str`): The model specification string for SEM in lavaan syntax.
    - `var_names` (`Optional[Dict[str, str]]`, default `None`): A dictionary mapping current factor names to meaningful names. Example: `{'Academic': 'Academic_Ability', 'Arts': 'Artistic_Skills'}`.
    - `estimator` (`str`, default `"MLM"`): The estimator to use for fitting the SEM model. Options include `"MLM"`, `"MLR"`, `"ULS"`, `"WLSMV"`.
    - `model_2_string` (`Optional[str]`, default `None`): The model specification string for the second SEM model to compare.
    - `ordered` (`Optional[List[str]]`, default `None`): A list of variable names that are ordered (ordinal variables).

- **Returns:**
    - `Dict[str, Any]`: A dictionary containing:
        - `'model_string'`: The SEM model specification string.
        - `'fit_summary'`: The SEM fit summary as a string.
        - `'fit_measures'`: Selected fit indices as a dictionary.
        - `'measurement_model'`: Parameter estimates for the measurement model.
        - `'structural_model'`: Parameter estimates for the structural model.
        - `'residual_covariances'`: Residual covariances.
        - `'factor_scores'`: Factor scores for each participant.
        - `'r2'`: R² values for endogenous variables.
        - `'log_likelihood'`: The total log-likelihood of the model (if available).
        - `'log_likelihoods'`: Per-sample log-likelihoods (if available).
        - `'npar'`: Number of parameters estimated in the model.
        - `'n_samples'`: Number of observations in the data.
        - `'comparison_results'`: Model comparison results (if `model_2_string` is provided).
        - `'is_better_model'`: Indicator of which model is better.
        - `'model_2_string'`: The second model specification string (if provided).

- **Usage Example:**

  ```python
  import pandas as pd
  
  data = pd.read_csv("your_data.csv")
  model_1 = """
  Academic =~ Math + Science + Literature
  Sports =~ Football + Basketball + Tennis
  Academic ~ Sports
  """
  
  results = fit_sem_lavaan(
      data=data,
      model_1_string=model_1,
      estimator="MLM",
      ordered=["Math", "Science", "Literature", "Football", "Basketball", "Tennis"]
  )
  
  print(results["fit_summary"])
  ```

### search_best_graph_climber

**`search_best_graph_climber`** searches for the best graph structure using hill-climbing based on SEM fit.

```python
search_best_graph_climber(
    data: pd.DataFrame,
    initial_graph: GeneralGraph,
    node_names: Optional[List[str]] = None,
    max_iter: int = 1000,
    estimator: str = "MLM",
    **kwargs,
) -> Tuple[GeneralGraph, Dict[str, Any]]
```

- **Parameters:**
  - `data` (`pd.DataFrame`): The dataset used for SEM fitting.
  - `initial_graph` (`GeneralGraph`): The initial graph structure to start the search.
  - `node_names` (`Optional[List[str]]`, default `None`): A list of variable names in the dataset.
  - `max_iter` (`int`, default `1000`): The maximum number of iterations for the hill-climbing search.
  - `estimator` (`str`, default `"MLM"`): The estimator to use for fitting the SEM model. Options include `"MLM"`, `"MLR"`, `"ULS"`, `"WLSMV"`.
  - `**kwargs`: Additional keyword arguments.

- **Returns:**
  - `Tuple[GeneralGraph, Dict[str, Any]]`: A tuple containing:
    - `best_graph` (`GeneralGraph`): The graph structure with the best SEM fit.
    - `best_score` (`Dict[str, Any]`): The SEM fitting results for the best graph.

- **Usage Example:**

  ```python
  import pandas as pd
  from causallearn.graph.GeneralGraph import GeneralGraph
  
  data = pd.read_csv("your_data.csv")
  initial_graph = GeneralGraph(nodes=[0, 1, 2], edges={})
  
  best_graph, best_score = search_best_graph_climber(
      data=data,
      initial_graph=initial_graph,
      max_iter=500,
      estimator="MLR"
  )
  
  print(best_graph)
  print(best_score["fit_measures"])
  ```
