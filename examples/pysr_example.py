"""Demonstration of using PySR for causal effect estimation."""

import numpy as np
import pandas as pd

from causal_pipe.background_knowledge import BackgroundKnowledge
from causal_pipe.causal_pipe import CausalPipe, CausalPipeConfig
from causal_pipe.pipe_config import (
    VariableTypes,
    FASSkeletonMethod,
    FCIOrientationMethod,
    PYSRCausalEffectMethod,
)
from causallearn.graph.GraphNode import GraphNode


def run_pysr_example() -> None:
    """Generate a small nonlinear dataset and fit PySR equations."""
    try:
        import pysr  # noqa: F401
    except ImportError:
        print("PySR example skipped: install pysr and Julia to run symbolic regression.")
        return

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.5 * x - 0.8 * z + x * z + rng.normal(0, 0.15, size=n)

    df = pd.DataFrame({"x": x, "z": z, "y": y})

    knowledge = BackgroundKnowledge()
    x_node, z_node, y_node = GraphNode("x"), GraphNode("z"), GraphNode("y")
    knowledge.add_required_by_node(x_node, y_node)
    knowledge.add_required_by_node(z_node, y_node)
    knowledge.add_forbidden_by_node(y_node, x_node)
    knowledge.add_forbidden_by_node(y_node, z_node)

    config = CausalPipeConfig(
        variable_types=VariableTypes(continuous=["x", "z", "y"]),
        skeleton_method=FASSkeletonMethod(),
        orientation_method=FCIOrientationMethod(background_knowledge=knowledge),
        causal_effect_methods=[
            PYSRCausalEffectMethod(
                pysr_params={
                    "niterations": 20,
                    "population_size": 30,
                    "binary_operators": ["+", "-", "*"],
                    "unary_operators": [],
                    "maxsize": 10,
                    "maxdepth": 3,
                    "verbosity": 0,
                    "progress": False,
                }
            )
        ],
        show_plots=False,
    )

    pipe = CausalPipe(config)
    pipe.run_pipeline(df)
    results = pipe.causal_effects

    if "pysr" not in results:
        print("PySR did not produce results. Pipeline errors:")
        pipe.show_errors()
        return

    print("PySR structural equations:")
    for var, info in results["pysr"]["structural_equations"].items():
        if var != "y":
            continue
        eq = info.get("equation")
        r2 = info.get("r2")
        print(f"{var} = {eq} (R^2={r2:.3f})")


if __name__ == "__main__":
    run_pysr_example()
