from causal_pipe.causal_pipe import (
    FCIOrientationMethod,
    FASSkeletonMethod,
    CausalPipeConfig,
)
from causal_pipe.pipe_config import (
    VariableTypes,
    DataPreprocessingParams,
    CausalEffectMethod,
)
from examples.easy import compare_easy_dataset
from examples.easy_ordinal import compare_easy_dataset_with_ordinal
from examples.hard import compare_hard_dataset

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
        handling_missing="impute",
        imputation_method="mice",
        use_r_mice=True,
        full_obs_cols=None,
    )

    config = CausalPipeConfig(
        variable_types=VariableTypes(continuous=[], ordinal=[], nominal=[]),
        preprocessing_params=preprocessor_params,
        skeleton_method=FASSkeletonMethod(),
        orientation_method=FCIOrientationMethod(fci_bootstrap_resamples=20),
        causal_effect_methods=[
            # Best method - Respect FCI Edge Directions - No Climbing
            CausalEffectMethod(name="sem", directed=True, params={"estimator": "MLR"}),
            # For ordinal data
            # CausalEffectMethod(
            #     name="sem", directed=True, params={"estimator": "WLSMV"}
            # ),
            # Simple pearson/spearman partial correlation
            CausalEffectMethod(name="pearson", directed=True),
            # CausalEffectMethod(name="spearman", directed=True),
            # SEM Climbing, only ML based estimators are supported
            CausalEffectMethod(
                name="sem-climbing", directed=True, params={"estimator": "ML", "respect_pag": True,
                                                            "hc_bootstrap_resamples": 2, "hc_bootstrap_random_state": 42}

            ),
            # CausalEffectMethod(
            #     name="sem-climbing", directed=True, params={"estimator": "MLR"}
            # ),
        ],
        study_name="pipe_easy_dataset",
        output_path="./output/",
        show_plots=False,
        verbose=True,
    )
    # compare_easy_dataset(config)
    compare_easy_dataset_with_ordinal(config)
    # compare_hard_dataset(config)
