"""Temporal panel mediation example: x lag-2 affects y through m lag-1."""

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
    TemporalConfig,
    VariableTypes,
)


def generate_temporal_panel_mediation(
    seed: int = 42,
    n_subjects: int = 40,
    n_timepoints: int = 8,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        trait = rng.normal(scale=0.4)
        x_hist = [rng.normal(), rng.normal()]
        m_hist = [rng.normal(), rng.normal()]
        y_hist = [rng.normal(), rng.normal()]
        for time in range(n_timepoints):
            x = 0.45 * x_hist[-1] + trait + rng.normal(scale=0.7)
            m = 0.55 * m_hist[-1] + 0.65 * x_hist[-1] + rng.normal(scale=0.7)
            y = 0.45 * y_hist[-1] + 0.75 * m_hist[-1] + rng.normal(scale=0.8)
            rows.append({"id": subject, "time": time, "x": x, "m": m, "y": y})
            x_hist.append(x)
            m_hist.append(m)
            y_hist.append(y)
    return pd.DataFrame(rows)


def build_config() -> CausalPipeConfig:
    return CausalPipeConfig(
        variable_types=VariableTypes(continuous=["x", "m", "y"]),
        preprocessing_params=DataPreprocessingParams(
            handling_missing=HandlingMissingEnum.DROP,
            cat_to_codes=False,
            standardize=True,
        ),
        skeleton_method=FASSkeletonMethod(
            depth=3,
            bootstrap_resamples=0,
            bootstrap_random_state=42,
        ),
        orientation_method=FCIOrientationMethod(max_path_length=3),
        causal_effect_methods=[PearsonCausalEffectMethod()],
        temporal_config=TemporalConfig(
            id_col="id",
            time_col="time",
            lags=[1, 2],
            within_person_center=True,
            force_autoregressive_edges=True,
        ),
        mediation_config=MediationAnalysisConfig(
            specs=[MediationSpec(treatment="x", mediators=["m"], outcome="y")],
            bootstrap_samples=50,
        ),
        study_name="mediation_temporal_panel",
        output_path="./output/mediation",
        show_plots=False,
    )


def main() -> None:
    pipe = CausalPipe(build_config())
    pipe.run_pipeline(generate_temporal_panel_mediation())
    results = pipe.run_mediation_analysis()
    analysis = results["analyses"]["1_x_m_y"]
    print("Resolved temporal path:", analysis["spec"]["resolved"])
    print("Classification:", analysis["classification"])
    print("Total indirect effect:", analysis["effects"]["total_indirect_effect"])


if __name__ == "__main__":
    main()
