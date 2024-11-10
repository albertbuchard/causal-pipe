import uuid
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import (
    BaseModel,
    Field,
    UUID4,
    validator,
    ValidationError,
    field_validator,
)

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


# Define Enums for various configurable options


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
    SPEARMAN = "spearman"
    LASSO = "lasso"


class SkeletonMethodNameEnum(str, Enum):
    BCSL = "BCSL"
    FAS = "FAS"


class ConditionalIndependenceMethodEnum(str, Enum):
    # From causal-learn
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


# Pydantic Models with Validations


class VariableTypes(BaseModel):
    """
    Define variable types for the dataset.
    """

    continuous: List[str]
    ordinal: List[str] = Field(default_factory=list)
    nominal: List[str] = Field(default_factory=list)


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
        keep_only_correlated_with (Optional[List[str]]): List of targets. Only features correlated with these targets are kept.
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

    # Optional: Additional validation if needed
    @field_validator("filter_threshold")
    @classmethod
    def check_filter_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("filter_threshold must be between 0.0 and 1.0")
        return v

    class Config:
        validate_assignment = True


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

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v

    class Config:
        validate_assignment = True


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

    @field_validator("num_bootstrap_samples")
    @classmethod
    def check_num_bootstrap_samples(cls, v):
        if v <= 0:
            raise ValueError("num_bootstrap_samples must be positive")
        return v

    @field_validator("use_aee_alpha", "alpha", mode="before")
    @classmethod
    def check_alpha_values(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("alpha values must be between 0.0 and 1.0")
        return v

    @field_validator("max_k")
    @classmethod
    def check_max_k(cls, v):
        if v < 0:
            raise ValueError("max_k must be non-negative")
        return v


class FASSkeletonMethod(SkeletonMethod):
    """
    Configuration for FAS skeleton identification.
    """

    name: SkeletonMethodNameEnum = SkeletonMethodNameEnum.FAS
    depth: int = 3
    knowledge: Optional[BackgroundKnowledge] = None

    @field_validator("depth")
    @classmethod
    def check_depth(cls, v):
        if v < 0:
            raise ValueError("depth must be non-negative")
        return v

    class Config:
        arbitrary_types_allowed = (
            True  # Allows non-Pydantic types like BackgroundKnowledge
        )


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


class FCIOrientationMethod(OrientationMethod):
    """
    Configuration for FCI orientation method.
    """

    name: OrientationMethodNameEnum = OrientationMethodNameEnum.FCI
    background_knowledge: Optional[BackgroundKnowledge] = None
    alpha: float = 0.05
    max_path_length: int = 3

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v

    @field_validator("max_path_length")
    @classmethod
    def check_max_path_length(cls, v):
        if v < 0:
            raise ValueError("max_path_length must be non-negative")
        return v

    class Config:
        arbitrary_types_allowed = (
            True  # Allows non-Pydantic types like BackgroundKnowledge
        )


class HillClimbingOrientationMethod(OrientationMethod):
    """
    Configuration for Hill Climbing orientation method.
    """

    name: OrientationMethodNameEnum = OrientationMethodNameEnum.HILL_CLIMBING
    max_k: int = 3
    multiple_comparison_correction: Optional[MultipleComparisonCorrectionEnum] = None

    @field_validator("max_k")
    @classmethod
    def check_max_k(cls, v):
        if v < 0:
            raise ValueError("max_k must be non-negative")
        return v


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


class CausalPipeConfig(BaseModel):
    """
    Comprehensive configuration for CausalPipe.

    Attributes:
        variable_types (VariableTypes): Definitions of variable types.
        preprocessing_params (DataPreprocessingParams): Data preprocessing parameters.
        skeleton_method (SkeletonMethod): Configuration for skeleton identification.
        orientation_method (OrientationMethod): Configuration for edge orientation.
        causal_effect_methods (List[CausalEffectMethod]): List of causal effect estimation methods.
        study_name (UUID4): Unique identifier for the study.
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
    seed: int = 42

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
