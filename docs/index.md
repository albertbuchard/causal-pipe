# CausalPipe Documentation

Welcome to the documentation for **CausalPipe**, a Python package that simplifies causal discovery and analysis. CausalPipe wraps around **Causal-Learn** and **Lavaan** to provide an easy-to-use, step-by-step process for performing causal analysis.

## Features
- **Data Preprocessing**: Handles missing data with multiple imputation, standardizes variables, and encodes categorical variables.
- **Causal Discovery**: Construct causal graphs using methods like Fast Adjacency Search (FAS) or Bootstrap-based Causal Structure Learning (BCSL).
- **Edge Orientation**: Use algorithms such as Fast Causal Inference (FCI) or Hill Climbing to orient edges in causal graphs.
- **Causal Effect Estimation**: Estimate effects using methods such as Partial Correlation, Structural Equation Modeling (SEM), or kernel methods.
- **Symbolic Regression**: Integrate [PySR](https://github.com/MilesCranmer/PySR) to learn nonlinear structural equations and score cyclic models via pseudo-likelihood or MMD<sup>2</sup>.
- **Temporal Data**: Build lag-expanded graphs for single time series or repeated-measures panel data using `TemporalConfig`.
- **Mediation Analysis**: Test user-defined direct, full-mediation, and partial-mediation paths with SEM-backed model comparison.
- **Visualization**: Visualize correlation graphs, causal graphs, and SEM results.

## Configuration Classes

CausalPipe is configured through a collection of dataclasses that define each
step of the pipeline:

- **VariableTypes** – declare continuous, ordinal, and nominal variables.
- **DataPreprocessingParams** – control imputation, standardization, and
  feature filtering.
- **Skeleton methods** – `FASSkeletonMethod`, `BCSLSkeletonMethod`.
- **Orientation methods** – `FCIOrientationMethod`,
  `HillClimbingOrientationMethod`.
- **Causal effect methods** – `PearsonCausalEffectMethod`,
  `SpearmanCausalEffectMethod`, `MICausalEffectMethod`,
  `KCICausalEffectMethod`, `SEMCausalEffectMethod`,
  `SEMClimbingCausalEffectMethod`, `PYSRCausalEffectMethod`,
  `PYSRCausalEffectMethodHillClimbing`.
- **Temporal mode** – `TemporalConfig` adds lag expansion, temporal
  background knowledge, and temporal bootstrap defaults for time-varying data.
- **Mediation analysis** – `MediationAnalysisConfig` and `MediationSpec`
  test hypothesized mediation paths after static or temporal preprocessing.

For longitudinal or time-series workflows, see [Temporal Data](temporal.md).
For direct/full/partial mediation testing, see [Mediation Analysis](mediation.md).
