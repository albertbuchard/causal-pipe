"""Easy temporal example: one lagged driver and one lagged outcome."""

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
    PearsonCausalEffectMethod,
    TemporalConfig,
    VariableTypes,
)


def generate_temporal_easy_panel(
    n_subjects: int = 80,
    n_timepoints: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate panel data with activity at t-1 causing mood at t."""

    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(n_subjects):
        subject_baseline = rng.normal(0.0, 0.4)
        activity_prev = rng.normal()
        mood_prev = rng.normal()
        for time in range(n_timepoints):
            activity = 0.55 * activity_prev + subject_baseline + rng.normal(0, 0.6)
            mood = 0.45 * mood_prev + 0.85 * activity_prev + rng.normal(0, 0.7)
            rows.append(
                {
                    "id": subject_id,
                    "time": time,
                    "activity": activity,
                    "mood": mood,
                }
            )
            activity_prev = activity
            mood_prev = mood

    return pd.DataFrame(rows)


def create_true_temporal_graph_easy() -> GeneralGraph:
    """Return the expected lag-expanded temporal graph for the easy example."""

    node_names = ["activity__lag1", "mood__lag1", "activity__t", "mood__t"]
    nodes = {name: GraphNode(name) for name in node_names}
    graph = GeneralGraph(list(nodes.values()))

    true_edges = [
        ("activity__lag1", "activity__t"),
        ("mood__lag1", "mood__t"),
        ("activity__lag1", "mood__t"),
    ]
    for source, target in true_edges:
        graph.add_edge(Edge(nodes[source], nodes[target], Endpoint.TAIL, Endpoint.ARROW))

    return graph


def build_temporal_easy_config(
    output_path: str = "./output/temporal",
    show_plots: bool = False,
    seed: int = 42,
) -> CausalPipeConfig:
    """Build a minimal temporal CausalPipe config for the easy example."""

    return CausalPipeConfig(
        variable_types=VariableTypes(continuous=["activity", "mood"]),
        preprocessing_params=DataPreprocessingParams(
            handling_missing=HandlingMissingEnum.DROP,
            cat_to_codes=False,
            standardize=True,
        ),
        skeleton_method=FASSkeletonMethod(
            conditional_independence_method=ConditionalIndependenceMethodEnum.FISHERZ,
            depth=2,
        ),
        orientation_method=FCIOrientationMethod(
            conditional_independence_method=ConditionalIndependenceMethodEnum.FISHERZ,
            max_path_length=2,
        ),
        causal_effect_methods=[PearsonCausalEffectMethod()],
        temporal_config=TemporalConfig(
            id_col="id",
            time_col="time",
            lags=[1],
            allow_contemporaneous_edges=True,
            force_autoregressive_edges=True,
            within_person_center=True,
        ),
        study_name="temporal_easy_dataset",
        output_path=output_path,
        show_plots=show_plots,
        verbose=True,
        seed=seed,
    )


def run_temporal_easy_example(config: Optional[CausalPipeConfig] = None) -> CausalPipe:
    """Run the easy temporal example and return the fitted pipeline."""

    config = config or build_temporal_easy_config()
    data = generate_temporal_easy_panel(seed=config.seed)

    true_graph = create_true_temporal_graph_easy()
    visualize_graph(
        true_graph,
        title="True Temporal Graph (EASY)",
        show=config.show_plots,
    )

    pipe = CausalPipe(config)
    pipe.run_pipeline(data)

    print("\nTemporal metadata:")
    print(pipe.temporal_metadata)
    print("\nExpected key relation: activity__lag1 -> mood__t")
    return pipe


if __name__ == "__main__":
    run_temporal_easy_example()
