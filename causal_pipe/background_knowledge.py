"""Compatibility wrapper for causal-learn background knowledge."""

try:
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
except Exception:
    class BackgroundKnowledge:
        """Small fallback used when tests stub out causal-learn internals."""

        def __init__(self):
            self.forbidden_edges = []
            self.required_edges = []

        def add_forbidden_by_pattern(self, source, target):
            self.forbidden_edges.append((source, target))

        def add_required_by_pattern(self, source, target):
            self.required_edges.append((source, target))
