# causal_pipe/config.py

import uuid
from dataclasses import field, dataclass
from typing import List, Optional, Dict, Any, Set

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


@dataclass
class VariableTypes:
    """
    Define variable types for the dataset.
    """

    continuous: List[str]
    ordinal: List[str] = field(default_factory=list)
    nominal: List[str] = field(default_factory=list)


@dataclass
class DataPreprocessingParams:
    """
    Parameters for data preprocessing.
    """

    no_preprocessing: bool = False

    handling_missing: str = "impute"  # Options could be 'impute', 'drop', etc.
    cat_to_codes: bool = True
    standardize: bool = True

    # Imputation parameters
    imputation_method: str = "mice"
    use_r_mice: bool = True
    full_obs_cols: Optional[List[str]] = None

    # Filter out features without correlation with the target
    keep_only_correlated_with: Optional[str] = None
    filter_method: str = "mi"
    filter_threshold: float = 0.1

    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SkeletonMethod:
    """
    Configuration for skeleton identification.
    """

    name: str
    conditional_independence_method: str = "fisherz"
    alpha: float = 0.05
    params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class BCSLSkeletonMethod(SkeletonMethod):
    """
    Configuration for BCSL skeleton identification.
    """

    name: str = "BCSL"
    num_bootstrap_samples: int = 100
    multiple_comparison_correction: str = "fdr"
    bootstrap_all_edges: bool = True
    use_aee_alpha: float = 0.05
    max_k: int = 3


@dataclass
class FASSkeletonMethod(SkeletonMethod):
    """
    Configuration for FAS skeleton identification.
    """

    name: str = "FAS"
    depth: int = 3
    knowledge: Optional[BackgroundKnowledge] = None


@dataclass
class OrientationMethod:
    """
    Configuration for edge orientation.
    """

    name: str = "FCI"  # Options: 'FCI', 'Hill Climbing'
    conditional_independence_method: str = "fisherz"


@dataclass
class FCIOrientationMethod(OrientationMethod):
    """
    Configuration for FCI orientation method.
    """

    name: str = "FCI"
    background_knowledge: Optional[BackgroundKnowledge] = None
    alpha: float = 0.05
    max_path_length: int = 3


@dataclass
class HillClimbingOrientationMethod(OrientationMethod):
    """
    Configuration for Hill Climbing orientation method.
    """

    name: str = "Hill Climbing"
    max_k: int = 3
    multiple_comparison_correction: str = "fdr"


@dataclass
class CausalEffectMethod:
    """
    Configuration for causal effect estimation methods.

    Parameters:
    ----------
    - name: str : Name of the method.
                        Options:
                        - 'pearson': Partial Pearson Correlation
                        - 'spearman': Partial Spearman Correlation
                        - 'mi': Conditional Mutual Information
                        - 'kci': Kernel Conditional Independence
                        - 'sem': Structural Equation Modeling
                        - 'sem-climbing': Structural Equation Modeling with Hill Climbing search of the best graph.
    - directed: bool : True if the method starts from the directed graph,
                                False if it will use the undirected graph (Markov Blanket / General Skeleton).
    - params: Optional[Dict[str, Any]] : Additional parameters for the method.
    """

    name: str = "pearson"
    directed: bool = True
    params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CausalPipeConfig:
    """
    Comprehensive configuration for CausalPipe.
    """

    variable_types: VariableTypes = field(
        default_factory=lambda: VariableTypes(continuous=[], ordinal=[], nominal=[])
    )
    preprocessing_params: DataPreprocessingParams = field(
        default_factory=DataPreprocessingParams
    )
    skeleton_method: SkeletonMethod = field(default_factory=FASSkeletonMethod)
    orientation_method: OrientationMethod = field(default_factory=FCIOrientationMethod)
    causal_effect_methods: List[CausalEffectMethod] = field(
        default_factory=lambda: [CausalEffectMethod()]
    )
    study_name: str = f"study_{uuid.uuid4()}"
    output_path: str = "./output/causal_toolkit_results"
    show_plots: bool = True
    verbose: bool = False
