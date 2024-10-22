
### Quick Start
```markdown
# Quick Start

This guide will help you get started with **CausalPipe** by showing how to set up a basic causal analysis pipeline.

## 1. Define Configuration

First, define the configuration for your causal discovery pipeline using the `CausalPipeConfig` dataclass. You’ll specify variable types, preprocessing parameters, and methods for skeleton identification and edge orientation.

```python
from causal_pipe.causal_pipe import (
    CausalPipeConfig, DataPreprocessingParams, FASSkeletonMethod, FCIOrientationMethod, CausalEffectMethod
)

preprocessor_params = DataPreprocessingParams(
    cat_to_codes=False,
    standardize=True,
    filter_method="mi",
    filter_threshold=0.1,
    handling_missing="impute",
    imputation_method="mice",
    use_r_mice=True
)

variable_types = {
    "continuous": ["age", "income"],
    "ordinal": ["education_level"],
    "nominal": ["gender", "diagnosis_1", "diagnosis_2"]
}

config = CausalPipeConfig(
    variable_types=variable_types,
    preprocessing_params=preprocessor_params,
    skeleton_method=FASSkeletonMethod(),
    orientation_method=FCIOrientationMethod(),
    causal_effect_methods=[CausalEffectMethod(name="pearson")],
    study_name="causal_analysis",
    output_path="./output",
    show_plots=True,
    verbose=True
)
```

## 2. Initialize CausalPipe

Initialize the CausalPipe toolkit by passing in the configuration:

```python
from causal_pipe.causal_pipe import CausalPipe

# Initialize CausalPipe
toolkit = CausalPipe(config)
```

## 3. Run the Causal Discovery Pipeline

Now, you can run the pipeline on your data:

```python
import pandas as pd

# Load your dataset
data = pd.read_csv("your_data.csv")

# Run the causal discovery pipeline
toolkit.run_pipeline(data)
```


---

For more details on the API, see the [API Reference](api_reference.md).

