"""Hard temporal example: two lags, latent traits, and multiple pathways."""

import numpy as np
import pandas as pd
from typing import Optional
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from causal_pipe import CausalPipe
from causal_pipe.pipe_config import (
    CausalPipeConfig,
    ConditionalIndependenceMethodEnum,
    DataPreprocessingParams,
    FASSkeletonMethod,
    FCIOrientationMethod,
    HandlingMissingEnum,
    PearsonCausalEffectMethod,
    SpearmanCausalEffectMethod,
    TemporalConfig,
    VariableTypes,
)

try:
    from .effect_reporting import print_causal_effect_summary, save_true_graph
except ImportError:
    from effect_reporting import print_causal_effect_summary, save_true_graph


def generate_temporal_hard_panel(
    n_subjects: int = 120,
    n_timepoints: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate panel data with latent subject traits and lag-1/lag-2 effects."""

    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(n_subjects):
        trait_risk = rng.normal(0.0, 0.7)
        trait_capacity = rng.normal(0.0, 0.5)

        x_hist = [rng.normal(), rng.normal()]
        load_hist = [rng.normal(), rng.normal()]
        recovery_hist = [rng.normal(), rng.normal()]
        stress_hist = [rng.normal(), rng.normal()]
        symptom_hist = [rng.normal(), rng.normal()]
        outcome_hist = [rng.normal(), rng.normal()]

        for time in range(n_timepoints):
            exposure = (
                0.45 * x_hist[-1]
                + 0.20 * x_hist[-2]
                + trait_risk
                + rng.normal(0, 0.7)
            )
            load = (
                0.50 * load_hist[-1]
                + 0.55 * exposure
                + 0.25 * x_hist[-1]
                + rng.normal(0, 0.7)
            )
            recovery = (
                0.40 * recovery_hist[-1]
                - 0.45 * load_hist[-1]
                + trait_capacity
                + rng.normal(0, 0.7)
            )
            stress = (
                0.45 * stress_hist[-1]
                + 0.65 * load_hist[-1]
                - 0.35 * recovery_hist[-1]
                + rng.normal(0, 0.8)
            )
            symptom = (
                0.50 * symptom_hist[-1]
                + 0.60 * stress_hist[-1]
                + 0.85 * load_hist[-2]
                + rng.normal(0, 0.8)
            )
            outcome = (
                0.45 * outcome_hist[-1]
                + 0.70 * symptom_hist[-1]
                - 0.75 * recovery_hist[-2]
                + rng.normal(0, 0.9)
            )

            rows.append(
                {
                    "id": subject_id,
                    "time": time,
                    "exposure": exposure,
                    "load": load,
                    "recovery": recovery,
                    "stress": stress,
                    "symptom": symptom,
                    "outcome": outcome,
                }
            )

            x_hist.append(exposure)
            load_hist.append(load)
            recovery_hist.append(recovery)
            stress_hist.append(stress)
            symptom_hist.append(symptom)
            outcome_hist.append(outcome)

    return pd.DataFrame(rows)


def create_true_temporal_graph_hard() -> GeneralGraph:
    """Return the expected lag-expanded temporal graph for the hard example."""

    variables = ["exposure", "load", "recovery", "stress", "symptom", "outcome"]
    node_names = (
        [f"{v}__lag2" for v in variables]
        + [f"{v}__lag1" for v in variables]
        + [f"{v}__t" for v in variables]
    )
    nodes = {name: GraphNode(name) for name in node_names}
    graph = GeneralGraph(list(nodes.values()))

    true_edges = [
        ("exposure__lag1", "exposure__t"),
        ("exposure__lag2", "exposure__t"),
        ("load__lag1", "load__t"),
        ("recovery__lag1", "recovery__t"),
        ("stress__lag1", "stress__t"),
        ("symptom__lag1", "symptom__t"),
        ("outcome__lag1", "outcome__t"),
        ("exposure__t", "load__t"),
        ("exposure__lag1", "load__t"),
        ("load__lag1", "recovery__t"),
        ("load__lag1", "stress__t"),
        ("recovery__lag1", "stress__t"),
        ("stress__lag1", "symptom__t"),
        ("load__lag2", "symptom__t"),
        ("symptom__lag1", "outcome__t"),
        ("recovery__lag2", "outcome__t"),
    ]
    for source, target in true_edges:
        graph.add_edge(Edge(nodes[source], nodes[target], Endpoint.TAIL, Endpoint.ARROW))

    return graph


def build_temporal_hard_config(
    output_path: str = "./output/temporal",
    show_plots: bool = False,
    seed: int = 42,
) -> CausalPipeConfig:
    """Build a stricter temporal config for a harder two-lag panel."""

    variables = ["exposure", "load", "recovery", "stress", "symptom", "outcome"]
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
            bootstrap_resamples=30,
            bootstrap_random_state=seed,
        ),
        orientation_method=FCIOrientationMethod(
            conditional_independence_method=ConditionalIndependenceMethodEnum.FISHERZ,
            max_path_length=4,
        ),
        causal_effect_methods=[
            PearsonCausalEffectMethod(),
            SpearmanCausalEffectMethod(),
        ],
        temporal_config=TemporalConfig(
            id_col="id",
            time_col="time",
            lags=[1, 2],
            variables=variables,
            allow_contemporaneous_edges=True,
            force_autoregressive_edges=True,
            within_person_center=True,
        ),
        study_name="temporal_hard_dataset",
        output_path=output_path,
        show_plots=show_plots,
        verbose=True,
        seed=seed,
    )


def run_temporal_hard_example(config: Optional[CausalPipeConfig] = None) -> CausalPipe:
    """Run the hard temporal example and return the fitted pipeline."""

    config = config or build_temporal_hard_config()
    data = generate_temporal_hard_panel(seed=config.seed)

    true_graph = create_true_temporal_graph_hard()
    save_true_graph(
        true_graph,
        title="True Temporal Graph (HARD)",
        output_path=config.output_path,
        filename="temporal_hard_true_graph.png",
    )

    pipe = CausalPipe(config)
    pipe.run_pipeline(data)

    print("\nTemporal metadata:")
    print(pipe.temporal_metadata)
    print_causal_effect_summary(
        pipe,
        expected_edges=[
            "load__lag2 -> load__lag1",
            "outcome__lag2 -> outcome__lag1",
            "recovery__lag2 -> stress__lag1",
            "stress__lag1 -> symptom__t",
            "symptom__lag1 -> outcome__t",
        ],
        top_n=12,
    )
    return pipe


if __name__ == "__main__":
    run_temporal_hard_example()
