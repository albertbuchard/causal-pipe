"""Helpers for running example files directly from the repository."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")


def ensure_repo_root_on_path() -> None:
    """Prefer this checkout over similarly named third-party packages."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def default_example_config():
    """Return a lightweight config suitable for standalone static examples."""

    ensure_repo_root_on_path()
    from causal_pipe.pipe_config import (
        CausalPipeConfig,
        ConditionalIndependenceMethodEnum,
        DataPreprocessingParams,
        FASSkeletonMethod,
        FCIOrientationMethod,
        HandlingMissingEnum,
        MICausalEffectMethod,
        PearsonCausalEffectMethod,
        SpearmanCausalEffectMethod,
        VariableTypes,
    )

    return CausalPipeConfig(
        variable_types=VariableTypes(continuous=[], ordinal=[], nominal=[]),
        preprocessing_params=DataPreprocessingParams(
            handling_missing=HandlingMissingEnum.DROP,
            cat_to_codes=True,
            standardize=True,
        ),
        skeleton_method=FASSkeletonMethod(
            conditional_independence_method=ConditionalIndependenceMethodEnum.FISHERZ,
            depth=2,
        ),
        orientation_method=FCIOrientationMethod(max_path_length=2),
        causal_effect_methods=[
            PearsonCausalEffectMethod(),
            SpearmanCausalEffectMethod(),
            MICausalEffectMethod(),
        ],
        output_path="./output/examples",
        show_plots=False,
        verbose=False,
    )


def example_plot_path(filename: str) -> str:
    """Return a non-interactive output path for example teaching figures."""

    output_dir = Path("./output/examples/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)
