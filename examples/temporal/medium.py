"""Medium temporal example: mediation with within-person dynamics."""

import numpy as np
import pandas as pd
from typing import Optional
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from causal_pipe import CausalPipe
from causal_pipe.causal_discovery.static_causal_discovery import visualize_graph
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
    TemporalConfig,
    VariableTypes,
)

try:
    from .effect_reporting import print_causal_effect_summary
except ImportError:
    from effect_reporting import print_causal_effect_summary


def generate_temporal_medium_panel(
    n_subjects: int = 100,
    n_timepoints: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate panel data with lagged activity -> stress -> performance."""

    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(n_subjects):
        trait = rng.normal(0.0, 0.6)
        activity_prev = rng.normal()
        stress_prev = rng.normal()
        sleep_prev = rng.normal()
        performance_prev = rng.normal()
        for time in range(n_timepoints):
            activity = 0.50 * activity_prev + trait + rng.normal(0, 0.7)
            sleep = 0.45 * sleep_prev - 0.35 * stress_prev + rng.normal(0, 0.7)
            stress = 0.55 * stress_prev - 0.60 * activity_prev + rng.normal(0, 0.7)
            performance = (
                0.40 * performance_prev
                + 0.55 * sleep_prev
                - 0.70 * stress_prev
                + rng.normal(0, 0.8)
            )
            rows.append(
                {
                    "id": subject_id,
                    "time": time,
                    "activity": activity,
                    "stress": stress,
                    "sleep": sleep,
                    "performance": performance,
                }
            )
            activity_prev = activity
            stress_prev = stress
            sleep_prev = sleep
            performance_prev = performance

    return pd.DataFrame(rows)


def create_true_temporal_graph_medium() -> GeneralGraph:
    """Return the expected lag-expanded temporal graph for the medium example."""

    variables = ["activity", "stress", "sleep", "performance"]
    node_names = [f"{v}__lag1" for v in variables] + [f"{v}__t" for v in variables]
    nodes = {name: GraphNode(name) for name in node_names}
    graph = GeneralGraph(list(nodes.values()))

    true_edges = [
        ("activity__lag1", "activity__t"),
        ("stress__lag1", "stress__t"),
        ("sleep__lag1", "sleep__t"),
        ("performance__lag1", "performance__t"),
        ("activity__lag1", "stress__t"),
        ("stress__lag1", "sleep__t"),
        ("sleep__lag1", "performance__t"),
        ("stress__lag1", "performance__t"),
    ]
    for source, target in true_edges:
        graph.add_edge(Edge(nodes[source], nodes[target], Endpoint.TAIL, Endpoint.ARROW))

    return graph


def build_temporal_medium_config(
    output_path: str = "./output/temporal",
    show_plots: bool = False,
    seed: int = 42,
) -> CausalPipeConfig:
    """Build a temporal config that keeps within-person dynamics central."""

    variables = ["activity", "stress", "sleep", "performance"]
    return CausalPipeConfig(
        variable_types=VariableTypes(continuous=variables),
        preprocessing_params=DataPreprocessingParams(
            handling_missing=HandlingMissingEnum.DROP,
            cat_to_codes=False,
            standardize=True,
        ),
        skeleton_method=FASSkeletonMethod(
            conditional_independence_method=ConditionalIndependenceMethodEnum.FISHERZ,
            depth=3,
            bootstrap_resamples=20,
            bootstrap_random_state=seed,
        ),
        orientation_method=FCIOrientationMethod(
            conditional_independence_method=ConditionalIndependenceMethodEnum.FISHERZ,
            max_path_length=3,
        ),
        causal_effect_methods=[
            PearsonCausalEffectMethod(),
            SpearmanCausalEffectMethod(),
            MICausalEffectMethod(),
        ],
        temporal_config=TemporalConfig(
            id_col="id",
            time_col="time",
            lags=[1],
            variables=variables,
            allow_contemporaneous_edges=True,
            force_autoregressive_edges=True,
            within_person_center=True,
        ),
        study_name="temporal_medium_dataset",
        output_path=output_path,
        show_plots=show_plots,
        verbose=True,
        seed=seed,
    )


def run_temporal_medium_example(config: Optional[CausalPipeConfig] = None) -> CausalPipe:
    """Run the medium temporal example and return the fitted pipeline."""

    config = config or build_temporal_medium_config()
    data = generate_temporal_medium_panel(seed=config.seed)

    true_graph = create_true_temporal_graph_medium()
    visualize_graph(
        true_graph,
        title="True Temporal Graph (MEDIUM)",
        show=config.show_plots,
    )

    pipe = CausalPipe(config)
    pipe.run_pipeline(data)

    print("\nTemporal metadata:")
    print(pipe.temporal_metadata)
    print_causal_effect_summary(
        pipe,
        expected_edges=[
            "activity__lag1 -> stress__t",
            "stress__lag1 -> sleep__t",
            "sleep__lag1 -> performance__t",
            "stress__lag1 -> performance__t",
        ],
        top_n=10,
    )
    return pipe


if __name__ == "__main__":
    run_temporal_medium_example()
