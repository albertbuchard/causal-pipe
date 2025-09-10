from causal_pipe.causal_pipe import (
    FCIOrientationMethod,
    FASSkeletonMethod,
    CausalPipeConfig,
)
from causal_pipe.pipe_config import (
    VariableTypes,
    DataPreprocessingParams,
    CausalEffectMethod, CausalEffectMethodNameEnum, PYSRCausalEffectMethod, PearsonCausalEffectMethod,
    SEMCausalEffectMethod, SEMClimbingCausalEffectMethod, SpearmanCausalEffectMethod,
    PYSRCausalEffectMethodHillClimbing, HandlingMissingEnum,
)
from examples.easy import compare_easy_dataset
from examples.easy_ordinal import compare_easy_dataset_with_ordinal
from examples.hard import compare_hard_dataset
from examples.super_basic import compare_super_basic_dataset

if __name__ == "__main__":
    """
    Compare the easy and hard datasets using Causal Pipe
    """

    preprocessor_params = DataPreprocessingParams(
        cat_to_codes=False,
        standardize=True,
        # keep_only_correlated_with=None,
        # filter_method="mi",
        # filter_threshold=0.1,
        handling_missing=HandlingMissingEnum.DROP,
        # handling_missing="impute",
        # imputation_method="mice",
        use_r_mice=True,
        full_obs_cols=None,
    )

    config = CausalPipeConfig(
        variable_types=VariableTypes(continuous=[], ordinal=[], nominal=[]),
        preprocessing_params=preprocessor_params,
        skeleton_method=FASSkeletonMethod(
            bootstrap_resamples=50,
            # n_jobs=4 or None if you want to use (all available cores - 1)
        ),
        orientation_method=FCIOrientationMethod(),
        causal_effect_methods=[
            # ML SEM - Respect Partial Ancestor Graph - No Climbing
            # SEMCausalEffectMethod(estimator="ML", respect_pag=True),
            # For ordinal data
            # SEMCausalEffectMethod(estimator="WLSMV"),
            # Simple pearson/spearman partial correlation
            # PearsonCausalEffectMethod(),
            # SpearmanCausalEffectMethod(),
            SEMClimbingCausalEffectMethod(estimator="ML",
                                          respect_pag=True,
                                          chain_orientation=True, # Next Causal Effect Method will use the best graph found here
                                          finalize_with_resid_covariances=False),
            # PySR-based Causal Effect estimation
            PYSRCausalEffectMethod(
                pysr_params={
                    "niterations": 100,
                    "population_size": 100,
                    "maxsize": 10,
                    "binary_operators": ["+", "-", "*", "/"],
                    "unary_operators": ["exp", "log", "sin", "cos", "inv"],
                },
            )
            # PYSRCausalEffectMethodHillClimbing()
        ],
        study_name="pipe_super_basic_dataset",
        output_path="./output/",
        show_plots=False,
        verbose=True,
    )
    # compare_super_basic_dataset(config)
    # compare_easy_dataset(config)
    # compare_easy_dataset_with_ordinal(config)
    compare_hard_dataset(config)
