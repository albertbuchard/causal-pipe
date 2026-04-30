# Temporal Examples

These examples show how to use `TemporalConfig` with long-format panel data.
Each row is one observation for one unit at one time point.

The examples deliberately keep the input variables simple:

- `id`: individual or panel-unit identifier.
- `time`: ordered time index.
- measured variables such as `activity`, `stress`, or `outcome`.

CausalPipe expands those measured variables into nodes such as:

- `activity__lag1`
- `stress__lag1`
- `outcome__t`

Run an example directly:

```bash
python -m examples.temporal.easy
python -m examples.temporal.medium
python -m examples.temporal.hard
```

All three examples use FAS + FCI because milestone 1 temporal support is built
around lag expansion plus temporal background knowledge.
