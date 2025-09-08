from causal_pipe.causal_pipe import (
    FCIOrientationMethod,
    FASSkeletonMethod,
    CausalPipeConfig,
)
from causal_pipe.pipe_config import (
    VariableTypes,
    DataPreprocessingParams,
    CausalEffectMethod, CausalEffectMethodNameEnum,
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
        skeleton_method=FASSkeletonMethod(
            bootstrap_resamples=50,
            # n_jobs=4 or None if you want to use (all available cores - 1)
        ),
        orientation_method=FCIOrientationMethod(),
        causal_effect_methods=[
            # Best method - Respect FCI Edge Directions - No Climbing
            CausalEffectMethod(name=CausalEffectMethodNameEnum.SEM, directed=True, params={"estimator": "MLR"}),
            # For ordinal data
            # CausalEffectMethod(
            #     name="sem", directed=True, params={"estimator": "WLSMV"}
            # ),
            # Simple pearson/spearman partial correlation
            CausalEffectMethod(name=CausalEffectMethodNameEnum.PEARSON, directed=True),
            # CausalEffectMethod(name="spearman", directed=True),
            # SEM Climbing, only ML based estimators are supported
            # CausalEffectMethod(
            #     name="sem-climbing", directed=True, params={"estimator": "ML",
            #                                                 "respect_pag": True,
            #                                                 "finalize_with_resid_covariances": True}
            # ),
            # CausalEffectMethod(
            #     name="sem-climbing", directed=True, params={"estimator": "MLR"}
            # ),
            # PySR-based Causal Effect estimation
            CausalEffectMethod(
                name=CausalEffectMethodNameEnum.PYSR,
                directed=True,
                params={"hc_orient_undirected_edges": False},
            ),
        ],
        study_name="pipe_easy_dataset",
        output_path="./output/",
        show_plots=False,
        verbose=True,
    )
    compare_easy_dataset(config)
    # compare_easy_dataset_with_ordinal(config)
    # compare_hard_dataset(config)
