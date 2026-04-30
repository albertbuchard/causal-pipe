"""Static full-mediation example: x affects y through m."""

import numpy as np
import pandas as pd

from causal_pipe import CausalPipe
from causal_pipe.pipe_config import (
    CausalPipeConfig,
    DataPreprocessingParams,
    FASSkeletonMethod,
    FCIOrientationMethod,
    HandlingMissingEnum,
    MediationAnalysisConfig,
    MediationSpec,
    PearsonCausalEffectMethod,
    VariableTypes,
)


def generate_static_full_mediation(seed: int = 42, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.80 * x + rng.normal(scale=0.55, size=n)
    y = 0.90 * m + rng.normal(scale=0.60, size=n)
    return pd.DataFrame({"x": x, "m": m, "y": y})


def build_config() -> CausalPipeConfig:
    return CausalPipeConfig(
        variable_types=VariableTypes(continuous=["x", "m", "y"]),
        preprocessing_params=DataPreprocessingParams(
            handling_missing=HandlingMissingEnum.DROP,
            cat_to_codes=False,
            standardize=True,
        ),
        skeleton_method=FASSkeletonMethod(depth=2),
        orientation_method=FCIOrientationMethod(max_path_length=2),
        causal_effect_methods=[PearsonCausalEffectMethod()],
        mediation_config=MediationAnalysisConfig(
            specs=[MediationSpec(treatment="x", mediators=["m"], outcome="y")],
            bootstrap_samples=100,
        ),
        study_name="mediation_static_full",
        output_path="./output/mediation",
        show_plots=False,
    )


def main() -> None:
    pipe = CausalPipe(build_config())
    pipe.run_pipeline(generate_static_full_mediation())
    results = pipe.run_mediation_analysis()
    analysis = results["analyses"]["1_x_m_y"]
    print("Classification:", analysis["classification"])
    print("Total indirect effect:", analysis["effects"]["total_indirect_effect"])


if __name__ == "__main__":
    main()
