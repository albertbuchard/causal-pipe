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

**`CausalPipeConfig`** is a Pydantic model that encapsulates all configuration parameters required to set up the **CausalPipe** pipeline. It leverages enums for various configurable options and includes validations to ensure the integrity of the configuration.

#### Attributes

- `variable_types` (`VariableTypes`): Defines the types of variables (`continuous`, `ordinal`, `nominal`) in the dataset.
- `preprocessing_params` (`DataPreprocessingParams`): Parameters for data preprocessing.
- `skeleton_method` (`SkeletonMethod`): Configuration for skeleton identification methods.
  - `BCSLSkeletonMethod`
  - `FASSkeletonMethod`
- `orientation_method` (`OrientationMethod`): Configuration for edge orientation methods.
  - `FCIOrientationMethod`
  - `HillClimbingOrientationMethod`
- `causal_effect_methods` (`Optional[List[CausalEffectMethod]]`): List of methods for estimating causal effects (e.g., `CausalEffectMethod`).
- `study_name` (`str`): Unique identifier for the study.
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

---

#### DataPreprocessingParams

**`DataPreprocessingParams`** configures how data preprocessing is handled in the pipeline.

- **Attributes:**
    - `no_preprocessing` (`bool`, default `False`): Whether to skip preprocessing.
    - `handling_missing` (`HandlingMissingEnum`, default `HandlingMissingEnum.IMPUTE`): Method to handle missing values.
      - **Options:**
        - `"impute"`
        - `"drop"`
        - `"error"`
    - `cat_to_codes` (`bool`, default `True`): Whether to convert categorical variables to numeric codes.
    - `standardize` (`bool`, default `True`): Whether to standardize continuous variables.
    - `imputation_method` (`ImputationMethodEnum`, default `ImputationMethodEnum.MICE`): Method for imputation.
      - **Options:**
        - `"mice"`
        - `"simple"`
    - `use_r_mice` (`bool`, default `True`): Whether to use R's `mice` for imputation.
    - `full_obs_cols` (`Optional[List[str]]`, default `None`): Columns to keep as fully observed.
    - `keep_only_correlated_with` (`Optional[List[str]]`, default `None`): List of target variables. Only features correlated with these targets are kept.
    - `filter_method` (`FilterMethodEnum`, default `FilterMethodEnum.MUTUAL_INFO`): Method to filter out features without correlation with the target.
      - **Options:**
        - `"mutual_info"`
        - `"pearson"`
        - `"lasso"`
    - `filter_threshold` (`float`, default `0.1`): Threshold for the filter method. Must be between `0.0` and `1.0`.
    - `kwargs` (`Optional[Dict[str, Any]]`, default `{}`): Additional keyword arguments.

- **Validations:**
    - `filter_threshold` must be between `0.0` and `1.0`.

---

#### SkeletonMethod

**`SkeletonMethod`** is a base Pydantic model for configuring skeleton identification methods.

- **Attributes:**
    - `name` (`SkeletonMethodNameEnum`): Name of the skeleton method.
      - **Options:**
        - `"BCSL"`
        - `"FAS"`
    - `conditional_independence_method` (`ConditionalIndependenceMethodEnum`, default `ConditionalIndependenceMethodEnum.FISHERZ`): Method for conditional independence testing.
      - **Options:**
        - `"fisherz"`
        - `"kci"`
        - `"d_separation"`
        - `"gsq"`
        - `"chisq"`
        - `"mc_fisherz"`
        - `"mv_fisherz"`
    - `alpha` (`float`, default `0.05`): Significance level for tests. Must be between `0.0` and `1.0`.
    - `params` (`Optional[Dict[str, Any]]`, default `{}`): Additional parameters.

- **Validations:**
    - `alpha` must be between `0.0` and `1.0`.

---

##### BCSLSkeletonMethod

**`BCSLSkeletonMethod`** configures the Bootstrap-based Causal Structure Learning (BCSL) method for skeleton identification.

- **Inherits From:** `SkeletonMethod`

- **Attributes:**
    - `name` (`SkeletonMethodNameEnum`, default `SkeletonMethodNameEnum.BCSL`): Name of the skeleton method.
    - `num_bootstrap_samples` (`int`, default `100`): Number of bootstrap samples. Must be positive.
    - `multiple_comparison_correction` (`Optional[MultipleComparisonCorrectionEnum]`, default `None`): Method for multiple comparison correction.
      - **Options:**
        - `"fdr"`
        - `"bonferroni"`
    - `bootstrap_all_edges` (`bool`, default `True`): Whether to bootstrap all edges.
    - `use_aee_alpha` (`float`, default `0.05`): Alpha level for AEE. Must be between `0.0` and `1.0`.
    - `max_k` (`int`, default `3`): Maximum conditioning set size. Must be non-negative.

- **Validations:**
    - `num_bootstrap_samples` must be positive.
    - `use_aee_alpha` must be between `0.0` and `1.0`.
    - `alpha` (inherited) must be between `0.0` and `1.0`.
    - `max_k` must be non-negative.

---

##### FASSkeletonMethod

**`FASSkeletonMethod`** configures the Fast Adjacency Search (FAS) method for skeleton identification.

- **Inherits From:** `SkeletonMethod`

- **Attributes:**
    - `name` (`SkeletonMethodNameEnum`, default `SkeletonMethodNameEnum.FAS`): Name of the skeleton method.
    - `depth` (`int`, default `3`): Depth parameter for FAS. Must be non-negative.
    - `knowledge` (`Optional[BackgroundKnowledge]`, default `None`): Background knowledge for FAS.

- **Validations:**
    - `depth` must be non-negative.

- **Configuration:**
    - `arbitrary_types_allowed = True` (Allows non-Pydantic types like `BackgroundKnowledge`)

---

#### OrientationMethod

**`OrientationMethod`** is a base Pydantic model for configuring edge orientation methods.

- **Attributes:**
    - `name` (`OrientationMethodNameEnum`): Name of the orientation method.
      - **Options:**
        - `"FCI"`
        - `"Hill Climbing"`
    - `conditional_independence_method` (`ConditionalIndependenceMethodEnum`, default `ConditionalIndependenceMethodEnum.FISHERZ`): Method for conditional independence testing.
      - **Options:**
        - `"fisherz"`
        - `"kci"`
        - `"d_separation"`
        - `"gsq"`
        - `"chisq"`
        - `"mc_fisherz"`
        - `"mv_fisherz"`

---

##### FCIOrientationMethod

**`FCIOrientationMethod`** configures the Fast Causal Inference (FCI) method for edge orientation.

- **Inherits From:** `OrientationMethod`

- **Attributes:**
    - `name` (`OrientationMethodNameEnum`, default `OrientationMethodNameEnum.FCI`): Name of the orientation method.
    - `background_knowledge` (`Optional[BackgroundKnowledge]`, default `None`): Background knowledge for FCI.
    - `alpha` (`float`, default `0.05`): Significance level for tests. Must be between `0.0` and `1.0`.
    - `max_path_length` (`int`, default `3`): Maximum path length for FCI. Must be non-negative.
    - `fci_bootstrap_resamples` (`int`, default `0`): If greater than `0`, run FCI on that many bootstrap samples to estimate edge orientation stability. The three most frequent bootstrapped graphs are saved under `fci_bootstrap/`.
    - `fci_bootstrap_random_state` (`Optional[int]`, default `None`): Seed for the FCI bootstrap resampling procedure.

- **Validations:**
    - `alpha` must be between `0.0` and `1.0`.
    - `max_path_length` must be non-negative.
    - `fci_bootstrap_resamples` must be non-negative.

- **Configuration:**
    - `arbitrary_types_allowed = True` (Allows non-Pydantic types like `BackgroundKnowledge`)

---

##### HillClimbingOrientationMethod

**`HillClimbingOrientationMethod`** configures the Hill Climbing method for edge orientation.

- **Inherits From:** `OrientationMethod`

- **Attributes:**
    - `name` (`OrientationMethodNameEnum`, default `OrientationMethodNameEnum.HILL_CLIMBING`): Name of the orientation method.
    - `max_k` (`int`, default `3`): Maximum conditioning set size. Must be non-negative.
    - `multiple_comparison_correction` (`Optional[MultipleComparisonCorrectionEnum]`, default `None`): Method for multiple comparison correction.
      - **Options:**
        - `"fdr"`
        - `"bonferroni"`

- **Validations:**
    - `max_k` must be non-negative.

---

#### CausalEffectMethod

**`CausalEffectMethod`** configures the methods used for estimating causal effects.

- **Attributes:**
    - `name` (`CausalEffectMethodNameEnum`, default `CausalEffectMethodNameEnum.PEARSON`): Name of the causal effect estimation method.
      - **Options:**
        - `'pearson'`: Partial Pearson Correlation
        - `'spearman'`: Partial Spearman Correlation
        - `'mi'`: Conditional Mutual Information
        - `'kci'`: Kernel Conditional Independence
        - `'sem'`: Structural Equation Modeling
        - `'sem-climbing'`: Structural Equation Modeling with Hill Climbing search of the best graph.
    - `directed` (`bool`, default `True`): Indicates if the method uses a directed graph.
    - `params` (`Optional[Dict[str, Any]]`, default `{}`): Additional parameters for the method.
    - `hc_bootstrap_resamples` (`int`, default `0`): Number of bootstrap resamples for SEM hill climbing to estimate edge orientation stability. When greater than `0`, the three most frequent bootstrapped graphs are saved under `sem_hc_bootstrap/`.
    - `hc_bootstrap_random_state` (`Optional[int]`, default `None`): Seed for the SEM hill-climb bootstrap procedure.

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

#### **`exhaustive_results(general_graph: GeneralGraph, compared_to_graph: Optional[GeneralGraph] = None) -> Dict[str, Any]`**

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

---

### search_best_graph_climber

**`search_best_graph_climber`** searches for the best graph structure using hill-climbing based on SEM fit.

```python
search_best_graph_climber(
    data: pd.DataFrame,
    initial_graph: GeneralGraph,
    node_names: Optional[List[str]] = None,
    max_iter: int = 1000,
    estimator: str = "MLM",
    finalize_with_resid_covariances: bool = False,
    mi_cutoff: float = 10.0,
    sepc_cutoff: float = 0.10,
    max_add: int = 5,
    delta_stop: float = 0.003,
    whitelist_pairs: Optional[pd.DataFrame] = None,
    forbid_pairs: Optional[pd.DataFrame] = None,
    same_occasion_regex: Optional[str] = None,
    *,
    respect_pag: bool = False,
    hc_bootstrap_resamples: int = 0,
    hc_bootstrap_random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[GeneralGraph, Dict[str, Any]]
```

- **Parameters:**
    - `data` (`pd.DataFrame`): The dataset used for SEM fitting.
    - `initial_graph` (`GeneralGraph`): The initial graph structure to start the search.
    - `node_names` (`Optional[List[str]]`, default `None`): A list of variable names in the dataset.
    - `max_iter` (`int`, default `1000`): The maximum number of iterations for the hill-climbing search.
    - `estimator` (`str`, default `"MLM"`): The estimator to use for fitting the SEM model. Options include `"MLM"`, `"MLR"`, `"ULS"`, `"WLSMV"`.
    - `finalize_with_resid_covariances` (`bool`, default `False`): If `True`, run a post-hoc stepwise residual covariance augmentation.
    - `mi_cutoff` (`float`, default `10.0`): Minimum modification index threshold for considering a covariance.
    - `sepc_cutoff` (`float`, default `0.10`): Minimum absolute `sepc.all` threshold.
    - `max_add` (`int`, default `5`): Maximum number of covariances to add.
    - `delta_stop` (`float`, default `0.003`): Minimum improvement required in fit indices to continue.
    - `whitelist_pairs` (`Optional[pd.DataFrame]`, default `None`): Optional whitelist of pairs (`lhs`, `rhs`).
    - `forbid_pairs` (`Optional[pd.DataFrame]`, default `None`): Optional blocklist of pairs.
    - `same_occasion_regex` (`Optional[str]`, default `None`): Regex enforcing same-occasion pairs unless whitelisted.
    - `respect_pag` (`bool`, default `False`): When `True`, the search preserves PAG marks (no change to ↔, →, —; only resolves circle endpoints consistent with PAG semantics).
    - `hc_bootstrap_resamples` (`int`, default `0`): If greater than `0`, run the SEM hill climber on bootstrap resamples to estimate orientation probabilities after hill climbing. The three most common bootstrapped graphs are stored in `sem_hc_bootstrap/`.
    - `hc_bootstrap_random_state` (`Optional[int]`, default `None`): Seed for the hill-climb bootstrap resampling procedure.
    - `**kwargs`: Additional keyword arguments.

- **Returns:**
    - `Tuple[GeneralGraph, Dict[str, Any]]`: A tuple containing:
        - `best_graph` (`GeneralGraph`): The graph structure with the best SEM fit.
        - `best_score` (`Dict[str, Any]`): SEM results. When residual covariance augmentation is enabled, this dictionary also includes `without_added_covariance_score` (the pre-augmentation score) and `resid_cov_aug` (details of the augmentation step). If hill-climb bootstrapping is requested, `hc_edge_orientation_probabilities` contains orientation probabilities after hill climbing.

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
      estimator="MLR",
      finalize_with_resid_covariances=True,
  )

  print(best_graph)
  print(best_score["fit_measures"])
  ```

---

## Updated Configuration Classes

Below are the updated Pydantic classes used for configuring the **CausalPipe** pipeline. These classes incorporate enums for various configurable options and include validations to ensure configuration integrity.

### Enumerations

```python
from enum import Enum

class HandlingMissingEnum(str, Enum):
    IMPUTE = "impute"
    DROP = "drop"
    ERROR = "error"

class ImputationMethodEnum(str, Enum):
    MICE = "mice"
    SIMPLE = "simple"

class FilterMethodEnum(str, Enum):
    MUTUAL_INFO = "mutual_info"
    PEARSON = "pearson"
    LASSO = "lasso"

class SkeletonMethodNameEnum(str, Enum):
    BCSL = "BCSL"
    FAS = "FAS"

class ConditionalIndependenceMethodEnum(str, Enum):
    FISHERZ = "fisherz"
    KCI = "kci"
    D_SEPARATION = "d_separation"
    GSQ = "gsq"
    CHISQ = "chisq"
    MC_FISHERZ = "mc_fisherz"
    MV_FISHERZ = "mv_fisherz"

class MultipleComparisonCorrectionEnum(str, Enum):
    FDR = "fdr"
    BONFERRONI = "bonferroni"

class OrientationMethodNameEnum(str, Enum):
    FCI = "FCI"
    HILL_CLIMBING = "Hill Climbing"

class CausalEffectMethodNameEnum(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    MI = "mi"
    KCI = "kci"
    SEM = "sem"
    SEM_CLIMBING = "sem-climbing"
```

### Pydantic Models with Validations

#### VariableTypes

```python
from pydantic import BaseModel, Field
from typing import List

class VariableTypes(BaseModel):
    """
    Define variable types for the dataset.
    """

    continuous: List[str]
    ordinal: List[str] = Field(default_factory=list)
    nominal: List[str] = Field(default_factory=list)
```

#### DataPreprocessingParams

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class DataPreprocessingParams(BaseModel):
    """
    Parameters for data preprocessing.

    Attributes:
        no_preprocessing (bool): True if no preprocessing is required.
        handling_missing (HandlingMissingEnum): Method to handle missing values.
        cat_to_codes (bool): True if categorical variables should be converted to codes.
        standardize (bool): True if the data should be standardized.
        imputation_method (ImputationMethodEnum): Method to impute missing values.
        use_r_mice (bool): True if R MICE should be used for imputation.
        full_obs_cols (Optional[List[str]]): Columns with full observations - row is dropped if any missing values.
        keep_only_correlated_with (Optional[List[str]]): List of target variables. Only features correlated with these targets are kept.
        filter_method (FilterMethodEnum): Method to filter out features without correlation with the target.
        filter_threshold (float): Threshold for the filter method.
        kwargs (Optional[Dict[str, Any]]): Additional parameters for the preprocessing.
    """

    no_preprocessing: bool = False
    handling_missing: HandlingMissingEnum = HandlingMissingEnum.IMPUTE
    cat_to_codes: bool = True
    standardize: bool = True

    # Imputation parameters
    imputation_method: ImputationMethodEnum = ImputationMethodEnum.MICE
    use_r_mice: bool = True
    full_obs_cols: Optional[List[str]] = None

    # Filter out features without correlation with the target
    keep_only_correlated_with: Optional[List[str]] = None
    filter_method: FilterMethodEnum = FilterMethodEnum.MUTUAL_INFO
    filter_threshold: float = 0.1

    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Validation for filter_threshold
    @field_validator("filter_threshold")
    @classmethod
    def check_filter_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("filter_threshold must be between 0.0 and 1.0")
        return v

    class Config:
        validate_assignment = True
```

#### SkeletonMethod

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any

class SkeletonMethod(BaseModel):
    """
    Configuration for skeleton identification.
    """

    name: SkeletonMethodNameEnum
    conditional_independence_method: ConditionalIndependenceMethodEnum = (
        ConditionalIndependenceMethodEnum.FISHERZ
    )
    alpha: float = 0.05
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Validation for alpha
    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v

    class Config:
        validate_assignment = True
```

##### BCSLSkeletonMethod

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any

class BCSLSkeletonMethod(SkeletonMethod):
    """
    Configuration for BCSL skeleton identification.
    """

    name: SkeletonMethodNameEnum = SkeletonMethodNameEnum.BCSL
    num_bootstrap_samples: int = 100
    multiple_comparison_correction: Optional[MultipleComparisonCorrectionEnum] = None
    bootstrap_all_edges: bool = True
    use_aee_alpha: float = 0.05
    max_k: int = 3

    # Validation for num_bootstrap_samples
    @field_validator("num_bootstrap_samples")
    @classmethod
    def check_num_bootstrap_samples(cls, v):
        if v <= 0:
            raise ValueError("num_bootstrap_samples must be positive")
        return v

    # Validation for use_aee_alpha and alpha
    @field_validator("use_aee_alpha", "alpha", mode="before")
    @classmethod
    def check_alpha_values(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("alpha values must be between 0.0 and 1.0")
        return v

    # Validation for max_k
    @field_validator("max_k")
    @classmethod
    def check_max_k(cls, v):
        if v < 0:
            raise ValueError("max_k must be non-negative")
        return v
```

##### FASSkeletonMethod

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

class FASSkeletonMethod(SkeletonMethod):
    """
    Configuration for FAS skeleton identification.
    """

    name: SkeletonMethodNameEnum = SkeletonMethodNameEnum.FAS
    depth: int = 3
    knowledge: Optional[BackgroundKnowledge] = None

    # Validation for depth
    @field_validator("depth")
    @classmethod
    def check_depth(cls, v):
        if v < 0:
            raise ValueError("depth must be non-negative")
        return v

    class Config:
        arbitrary_types_allowed = True  # Allows non-Pydantic types like BackgroundKnowledge
```

---

#### OrientationMethod

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class OrientationMethod(BaseModel):
    """
    Configuration for edge orientation.
    """

    name: OrientationMethodNameEnum
    conditional_independence_method: ConditionalIndependenceMethodEnum = (
        ConditionalIndependenceMethodEnum.FISHERZ
    )

    class Config:
        validate_assignment = True
```

##### FCIOrientationMethod

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

class FCIOrientationMethod(OrientationMethod):
    """
    Configuration for FCI orientation method.
    """

    name: OrientationMethodNameEnum = OrientationMethodNameEnum.FCI
    background_knowledge: Optional[BackgroundKnowledge] = None
    alpha: float = 0.05
    max_path_length: int = 3
    fci_bootstrap_resamples: int = 0
    fci_bootstrap_random_state: Optional[int] = None

    # Validation for alpha
    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v

    # Validation for max_path_length
    @field_validator("max_path_length")
    @classmethod
    def check_max_path_length(cls, v):
        if v < 0:
            raise ValueError("max_path_length must be non-negative")
        return v

    @field_validator("fci_bootstrap_resamples")
    @classmethod
    def check_fci_bootstrap_resamples(cls, v):
        if v < 0:
            raise ValueError("fci_bootstrap_resamples must be non-negative")
        return v

    class Config:
        arbitrary_types_allowed = True  # Allows non-Pydantic types like BackgroundKnowledge
```

##### HillClimbingOrientationMethod

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any

class HillClimbingOrientationMethod(OrientationMethod):
    """
    Configuration for Hill Climbing orientation method.
    """

    name: OrientationMethodNameEnum = OrientationMethodNameEnum.HILL_CLIMBING
    max_k: int = 3
    multiple_comparison_correction: Optional[MultipleComparisonCorrectionEnum] = None

    # Validation for max_k
    @field_validator("max_k")
    @classmethod
    def check_max_k(cls, v):
        if v < 0:
            raise ValueError("max_k must be non-negative")
        return v
```

---

#### CausalEffectMethod

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class CausalEffectMethod(BaseModel):
    """
    Configuration for causal effect estimation methods.

    Attributes:
        name (CausalEffectMethodNameEnum): Name of the method.
        directed (bool): True if the method starts from the directed graph,
                        False if it will use the undirected graph (Markov Blanket / General Skeleton).
        params (Optional[Dict[str, Any]]): Additional parameters for the method.
    """

    name: CausalEffectMethodNameEnum = CausalEffectMethodNameEnum.PEARSON
    directed: bool = True
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
```

---

#### CausalPipeConfig

```python
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class CausalPipeConfig(BaseModel):
    """
    Comprehensive configuration for CausalPipe.

    Attributes:
        variable_types (VariableTypes): Definitions of variable types.
        preprocessing_params (DataPreprocessingParams): Data preprocessing parameters.
        skeleton_method (SkeletonMethod): Configuration for skeleton identification.
        orientation_method (OrientationMethod): Configuration for edge orientation.
        causal_effect_methods (Optional[List[CausalEffectMethod]]): List of causal effect estimation methods.
        study_name (str): Unique identifier for the study.
        output_path (str): Path to save the results.
        show_plots (bool): Whether to display plots.
        verbose (bool): Whether to enable verbose logging.
    """

    variable_types: VariableTypes = Field(
        default_factory=lambda: VariableTypes(continuous=[], ordinal=[], nominal=[])
    )
    preprocessing_params: DataPreprocessingParams = Field(
        default_factory=DataPreprocessingParams
    )
    skeleton_method: SkeletonMethod = Field(default_factory=FASSkeletonMethod)
    orientation_method: OrientationMethod = Field(default_factory=FCIOrientationMethod)
    causal_effect_methods: Optional[List[CausalEffectMethod]] = Field(
        default_factory=lambda: [CausalEffectMethod()]
    )
    study_name: str = Field(default_factory=lambda: f"study_{uuid.uuid4()}")
    output_path: str = "./output/causal_toolkit_results"
    show_plots: bool = True
    verbose: bool = False

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
```

---

### Enumerations Explained

- **`HandlingMissingEnum`**: Defines methods to handle missing data.
  - `"impute"`: Fill missing values using imputation.
  - `"drop"`: Remove rows with missing values.
  - `"error"`: Raise an error if missing values are found.

- **`ImputationMethodEnum`**: Specifies the imputation technique.
  - `"mice"`: Multivariate Imputation by Chained Equations.
  - `"simple"`: Simple imputation methods like mean or median.

- **`FilterMethodEnum`**: Determines the method for feature filtering based on correlation.
  - `"mutual_info"`: Mutual Information.
  - `"pearson"`: Pearson Correlation.
  - `"lasso"`: Lasso Regression for feature selection.

- **`SkeletonMethodNameEnum`**: Names of skeleton identification methods.
  - `"BCSL"`: Bootstrap-based Causal Structure Learning.
  - `"FAS"`: Fast Adjacency Search.

- **`ConditionalIndependenceMethodEnum`**: Methods used for conditional independence testing.
  - `"fisherz"`, `"kci"`, `"d_separation"`, `"gsq"`, `"chisq"`, `"mc_fisherz"`, `"mv_fisherz"`.

- **`MultipleComparisonCorrectionEnum`**: Techniques to correct for multiple comparisons.
  - `"fdr"`: False Discovery Rate.
  - `"bonferroni"`: Bonferroni Correction.

- **`OrientationMethodNameEnum`**: Names of edge orientation methods.
  - `"FCI"`: Fast Causal Inference.
  - `"Hill Climbing"`: Hill Climbing Algorithm.

- **`CausalEffectMethodNameEnum`**: Names of methods for estimating causal effects.
  - `'pearson'`, `'spearman'`, `'mi'`, `'kci'`, `'sem'`, `'sem-climbing'`.
