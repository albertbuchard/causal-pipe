# Mediation Analysis

Mediation analysis tests a user-defined causal path, such as `x -> m -> y`.
It does not replace causal discovery. Discovery can support the path and provide
graph-conditioned effect evidence, but the mediation hypothesis is supplied by
the user.

## Static Mediation

```python
from causal_pipe import CausalPipe
from causal_pipe.pipe_config import (
    CausalPipeConfig,
    MediationAnalysisConfig,
    MediationSpec,
    VariableTypes,
)

config = CausalPipeConfig(
    variable_types=VariableTypes(continuous=["x", "m", "y"]),
    mediation_config=MediationAnalysisConfig(
        specs=[
            MediationSpec(
                treatment="x",
                mediators=["m"],
                outcome="y",
            )
        ],
        bootstrap_samples=1000,
    ),
    show_plots=False,
)

pipe = CausalPipe(config)
pipe.run_pipeline(df)
results = pipe.run_mediation_analysis()
```

CausalPipe fits and compares three SEM models:

- `direct_only`: `x -> y`, while retaining the mediator model.
- `full_mediation`: `x -> m -> y`, without the direct `x -> y` path.
- `partial_mediation`: `x -> m -> y` plus direct `x -> y`.

The result contains:

- `classification`: `direct_only`, `full_mediation`, `partial_mediation`, or
  `no_clear_mediation`.
- `effects`: direct effect, indirect effects, total indirect effect, total
  effect, and proportion mediated when available.
- `model_comparison`: AIC, BIC, CFI, RMSEA, SRMR, R2, and selected best model.
- `graph_evidence`: whether requested path edges were present in the discovered
  graph, plus optional effect-method values.
- `warnings`: missing graph paths, temporal fallback notes, or failed model fits.

Outputs are written under:

```text
<output_path>/<study_name>/mediation/<spec_name>/
```

including `mediation_results.json`, `model_comparison.csv`,
`path_coefficients.csv`, and `mediation_summary.txt`.

## Multiple Mediators

Parallel mediation tests mediators independently:

```python
MediationSpec(
    treatment="x",
    mediators=["m1", "m2"],
    outcome="y",
    mode="parallel",
)
```

Serial mediation tests an ordered chain:

```python
MediationSpec(
    treatment="x",
    mediators=["m1", "m2"],
    outcome="y",
    mode="serial",
)
```

## Temporal Mediation

When `TemporalConfig` is enabled, mediation specs use original semantic names by
default. CausalPipe resolves those names to lag-expanded columns.

```python
from causal_pipe.pipe_config import TemporalConfig

config = CausalPipeConfig(
    variable_types=VariableTypes(continuous=["x", "m", "y"]),
    temporal_config=TemporalConfig(
        id_col="id",
        time_col="time",
        lags=[1, 2],
        within_person_center=True,
    ),
    mediation_config=MediationAnalysisConfig(
        specs=[MediationSpec(treatment="x", mediators=["m"], outcome="y")]
    ),
)
```

Temporal defaults:

- If lag 2 exists, the path resolves to `x__lag2 -> m__lag1 -> y__t`.
- If only lag 1 exists, the path resolves to `x__lag1 -> m__t -> y__t` and the
  result warns that mediator and outcome are same-wave.

Override temporal mapping explicitly:

```python
from causal_pipe.pipe_config import MediationTemporalLags

MediationSpec(
    treatment="x",
    mediators=["m"],
    outcome="y",
    temporal_lags=MediationTemporalLags(
        treatment=2,
        mediators=[1],
        outcome=0,
    ),
)
```

If you already want to reference lag-expanded columns directly, set
`variables_are_lagged=True`.

## Classification Rules

The SEM backend uses lavaan-defined parameters and bootstrap confidence
intervals when `bootstrap_samples > 0`.

- `partial_mediation`: indirect and direct effects are both supported, and the
  partial model is not worse than the full model.
- `full_mediation`: indirect effect is supported and the direct effect is not
  supported, or the full model is preferred/equivalent.
- `direct_only`: direct effect is supported and the indirect effect is not.
- `no_clear_mediation`: model fit failed, required effects are missing, or the
  evidence is contradictory.

Set `require_discovered_path=True` on a `MediationSpec` when the discovered
graph must contain the requested mediation path before the result can be called
mediation.
