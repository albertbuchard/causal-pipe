"""Demonstration of using PySR for causal effect estimation."""

import numpy as np
import pandas as pd

from causal_pipe.causal_pipe import CausalPipe, CausalPipeConfig
from causal_pipe.pipe_config import (
    VariableTypes,
    FASSkeletonMethod,
    FCIOrientationMethod,
    PYSRCausalEffectMethod,
)


def run_pysr_example() -> None:
    """Generate a small nonlinear dataset and fit PySR equations."""
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.sin(x) + z ** 2 + rng.normal(size=n)

    df = pd.DataFrame({"x": x, "z": z, "y": y})

    config = CausalPipeConfig(
        variable_types=VariableTypes(continuous=["x", "z", "y"]),
        skeleton_method=FASSkeletonMethod(),
        orientation_method=FCIOrientationMethod(),
        causal_effect_methods=[
            PYSRCausalEffectMethod(hc_orient_undirected_edges=True)
        ],
        show_plots=False,
    )

    pipe = CausalPipe(config)
    results = pipe.run_pipeline(df)

    print("PySR structural equations:")
    for var, info in results["pysr"]["structural_equations"].items():
        eq = info.get("equation")
        r2 = info.get("r2")
        print(f"{var} = {eq} (R^2={r2:.3f})")


if __name__ == "__main__":
    run_pysr_example()

