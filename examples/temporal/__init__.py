"""Temporal example datasets for CausalPipe."""

from examples.temporal.easy import (
    build_temporal_easy_config,
    create_true_temporal_graph_easy,
    generate_temporal_easy_panel,
    run_temporal_easy_example,
)
from examples.temporal.medium import (
    build_temporal_medium_config,
    create_true_temporal_graph_medium,
    generate_temporal_medium_panel,
    run_temporal_medium_example,
)
from examples.temporal.hard import (
    build_temporal_hard_config,
    create_true_temporal_graph_hard,
    generate_temporal_hard_panel,
    run_temporal_hard_example,
)

__all__ = [
    "build_temporal_easy_config",
    "create_true_temporal_graph_easy",
    "generate_temporal_easy_panel",
    "run_temporal_easy_example",
    "build_temporal_medium_config",
    "create_true_temporal_graph_medium",
    "generate_temporal_medium_panel",
    "run_temporal_medium_example",
    "build_temporal_hard_config",
    "create_true_temporal_graph_hard",
    "generate_temporal_hard_panel",
    "run_temporal_hard_example",
]
