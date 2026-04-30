"""Temporal example datasets for CausalPipe."""

from importlib import import_module


_EXPORTS = {
    "build_temporal_easy_config": "examples.temporal.easy",
    "create_true_temporal_graph_easy": "examples.temporal.easy",
    "generate_temporal_easy_panel": "examples.temporal.easy",
    "run_temporal_easy_example": "examples.temporal.easy",
    "build_temporal_medium_config": "examples.temporal.medium",
    "create_true_temporal_graph_medium": "examples.temporal.medium",
    "generate_temporal_medium_panel": "examples.temporal.medium",
    "run_temporal_medium_example": "examples.temporal.medium",
    "build_temporal_hard_config": "examples.temporal.hard",
    "create_true_temporal_graph_hard": "examples.temporal.hard",
    "generate_temporal_hard_panel": "examples.temporal.hard",
    "run_temporal_hard_example": "examples.temporal.hard",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
