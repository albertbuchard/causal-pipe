# Temporal Data

`CausalPipe` supports two modes:

- **Cross-sectional mode**: one row is one independent observation.
- **Temporal mode**: rows are ordered in time, optionally nested within an
  individual, subject, device, site, or other unit.

Temporal mode is enabled by adding `TemporalConfig` to `CausalPipeConfig`. The
pipeline converts long-format data into lag-expanded columns such as `x__t`,
`x__lag1`, and `x__lag2`, then runs the usual FAS, FCI, SEM, partial
correlation, or PySR downstream steps on those generated nodes.

## When To Use Each Mode

Use cross-sectional mode when rows can be treated as independent samples and
there is no meaningful row order.

Use temporal mode when a variable at an earlier observation can influence a
later observation. This includes:

- one time series, such as daily measurements from one process.
- repeated-measures panel data, such as many people measured over time.
- longitudinal survey data with waves or visits.

Do not add `time` as an ordinary causal variable if your goal is temporal
causality. Use `TemporalConfig` so the pipeline can create lagged predictors
and forbid impossible future-to-past edges.

## Cross-Sectional Setup

No temporal configuration is needed:

```python
from causal_pipe import CausalPipe
from causal_pipe.pipe_config import (
    CausalPipeConfig,
    DataPreprocessingParams,
    FASSkeletonMethod,
    FCIOrientationMethod,
    PearsonCausalEffectMethod,
    VariableTypes,
)

config = CausalPipeConfig(
    variable_types=VariableTypes(
        continuous=["age", "income", "symptom_score"],
        ordinal=[],
        nominal=["group"],
    ),
    preprocessing_params=DataPreprocessingParams(
        handling_missing="drop",
        cat_to_codes=True,
        standardize=True,
    ),
    skeleton_method=FASSkeletonMethod(),
    orientation_method=FCIOrientationMethod(),
    causal_effect_methods=[PearsonCausalEffectMethod()],
    show_plots=False,
)

pipe = CausalPipe(config)
pipe.run_pipeline(df)
effects = pipe.causal_effects
```

## Single Time-Series Setup

For one ordered series, provide only `time_col`.

```python
from causal_pipe.pipe_config import TemporalConfig

config = CausalPipeConfig(
    variable_types=VariableTypes(
        continuous=["stress", "sleep", "mood"],
    ),
    skeleton_method=FASSkeletonMethod(
        conditional_independence_method="fisherz",
        bootstrap_resamples=100,
    ),
    orientation_method=FCIOrientationMethod(),
    causal_effect_methods=[PearsonCausalEffectMethod()],
    temporal_config=TemporalConfig(
        time_col="day",
        lags=[1, 2],
        allow_contemporaneous_edges=True,
    ),
    show_plots=False,
)

pipe = CausalPipe(config)
pipe.run_pipeline(df)
```

Expected input shape:

| day | stress | sleep | mood |
| --- | --- | --- | --- |
| 1 | 0.4 | 7.0 | 0.2 |
| 2 | 0.6 | 6.5 | 0.1 |
| 3 | 0.7 | 6.0 | -0.1 |

Generated discovery columns include:

- `stress__t`
- `stress__lag1`
- `stress__lag2`
- `sleep__t`
- `sleep__lag1`
- `mood__t`

For a single series, FAS bootstrap defaults to moving block bootstrap so row
order is not destroyed.

## Repeated-Measures Panel Setup

For panel data, provide `id_col` and `time_col`. Lag construction never crosses
individual boundaries.

```python
config = CausalPipeConfig(
    variable_types=VariableTypes(
        continuous=["stress", "sleep", "mood"],
    ),
    skeleton_method=FASSkeletonMethod(
        conditional_independence_method="fisherz",
        bootstrap_resamples=100,
    ),
    orientation_method=FCIOrientationMethod(),
    causal_effect_methods=[PearsonCausalEffectMethod()],
    temporal_config=TemporalConfig(
        id_col="person_id",
        time_col="week",
        lags=[1],
        within_person_center=True,
        include_between_person_means=False,
    ),
    show_plots=False,
)

pipe = CausalPipe(config)
pipe.run_pipeline(df)
```

Expected input shape:

| person_id | week | stress | sleep | mood |
| --- | --- | --- | --- | --- |
| A | 1 | 0.4 | 7.0 | 0.2 |
| A | 2 | 0.6 | 6.5 | 0.1 |
| B | 1 | 0.2 | 8.0 | 0.3 |
| B | 2 | 0.3 | 7.5 | 0.2 |

For panel data, FAS bootstrap defaults to cluster bootstrap over individuals.
That keeps all generated observations from a sampled person together.

## TemporalConfig Reference

| Field | Meaning |
| --- | --- |
| `time_col` | Column used to sort observations within the series or person. |
| `id_col` | Optional subject/unit column. Required for panel data. |
| `lags` | Positive integer lags to generate. Default is `[1]`. |
| `variables` | Optional subset of declared variables to expand. Defaults to all variables in `VariableTypes`. |
| `allow_contemporaneous_edges` | If `True`, edges among `__t` variables are allowed. |
| `force_autoregressive_edges` | If `True`, require `x__lag1 -> x__t` for each variable. |
| `drop_rows_with_incomplete_lags` | If `True`, drop rows that lack requested lag values. |
| `within_person_center` | If `True`, subtract each person's mean before lag expansion. Requires `id_col` and continuous variables. |
| `include_between_person_means` | If `True`, add columns like `x__between` for person means. Requires `id_col`. |
| `bootstrap_unit` | Optional override: `row`, `block`, or `cluster`. |

## Temporal Background Knowledge

Temporal mode creates causal discovery constraints automatically:

- current-time nodes cannot cause lagged nodes.
- smaller lag distance cannot cause larger lag distance, for example
  `x__lag1 -> y__lag2` is forbidden.
- lagged nodes may cause current nodes.
- current-current edges are controlled by `allow_contemporaneous_edges`.
- autoregressive edges are required when `force_autoregressive_edges=True`.

If you also provide FAS or FCI background knowledge, CausalPipe merges temporal
constraints into it. User-provided constraints win if there is a conflict, and
conflicts are recorded in `temporal_metadata.json`.

## Individual Correlations Over Time

Repeated measures mix several sources of association:

- stable between-person differences.
- within-person deviations over time.
- autocorrelation.
- same-time residual correlation.

Use `within_person_center=True` when you want edges to represent within-person
dynamics. Use `include_between_person_means=True` when stable person-level
differences should remain visible as separate predictors.

For many longitudinal questions, this lag-expanded approach is a practical
first step. It is not a replacement for every dedicated temporal causal method.

## Outputs

Temporal runs write `temporal_metadata.json` in the study output folder. It
contains:

- original temporal columns.
- selected variables and lags.
- generated lagged columns.
- dropped-row counts.
- subject count for panel data.
- time range.
- background-knowledge summary.
- bootstrap mode.

Graphs use generated node names directly, for example `sleep__lag1 -> mood__t`.

## Limitations In This Milestone

- Temporal skeleton discovery supports FAS + FCI only.
- BCSL temporal mode raises `NotImplementedError`.
- PCMCI, dynamic Bayesian networks, and dedicated VAR-LiNGAM pipeline support
  are not part of this milestone.
- Lagged nodes are displayed as ordinary graph nodes.
