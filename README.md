# CausalPipe

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
![PyPI Version](https://img.shields.io/pypi/v/causal-pipe.svg)

**CausalPipe** is a Python wrapper built on [Causal-Learn](https://github.com/cmu-phil/causal-learn) and [Lavaan](https://lavaan.ugent.be/) that offers a predefined and well-formalized process for causal analysis tailored for everyday users. It provides intuitive tools for data preparation, constructing and orienting causal graphs, and visualizing results, supporting both ordinal and continuous variables.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Configuration](#1-configuration)
  - [2. Initializing CausalPipe](#2-initializing-causalpipe)
  - [3. Running the Causal Discovery Pipeline](#3-running-the-causal-discovery-pipeline)
- [Usage Examples](#usage-examples)
  - [Example: Running the Full Pipeline](#example-running-the-full-pipeline)
  - [Example: Custom Configuration](#example-custom-configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Data Preprocessing:** Handle missing values using multiple imputation (`MICE`), encode categorical variables, standardize features, and perform feature selection based on correlation.
- **Skeleton Identification:** Identify the global skeleton of the causal graph using methods like Fast Adjacency Search (`FAS`) or Bootstrap-based Causal Structure Learning (`BCSL`).
- **Edge Orientation:** Orient edges in the skeleton using algorithms such as Fast Causal Inference (`FCI`) or Hill Climbing.
- **Causal Effect Estimation:** Estimate causal effects using various methods, including Partial Pearson Correlation, Partial Spearman Correlation, Conditional Mutual Information (`MI`), Kernel Conditional Independence (`KCI`), Structural Equation Modeling (`SEM`), and Hill Climbing-based SEM.
- **Visualization:** Generate and save visualizations for correlation graphs, skeletons, oriented graphs, and SEM results.
- **Modular Configuration:** Easily configure different aspects of the pipeline through dataclasses, allowing for flexible and customizable causal discovery workflows.
- **Integration with R:** Utilize R's `lavaan` package for advanced Structural Equation Modeling directly within Python using `rpy2`.

## Installation

You can install `causal-pipe` via [PyPI](https://pypi.org/project/causal-pipe/) using `pip`:

```bash
pip install causal-pipe
```

### Dependencies

CausalPipe relies on several Python and R packages. Ensure that you have the following dependencies installed:

- **Python 3.6 or higher**
- **R:** Required for Structural Equation Modeling (`lavaan`) and multiple imputation (`mice`).
- **Python Packages:**
  - `numpy>=1.18.0`
  - `scipy>=1.4.0`
  - `scikit-learn>=0.22.0`
  - `causal-learn==0.1.3.8`
  - `bcsl-python==0.8.0`
  - `rpy2==3.5.16`
  - `npeet-plus==0.2.0`
  - `networkx==3.2.1`
  - `pandas==2.2.3`
  - `factor_analyzer==0.5.1`

## Quick Start

### 1. Configuration

Begin by defining the configuration for your causal discovery pipeline using the `CausalPipeConfig` dataclass. This includes specifying variable types, preprocessing parameters, skeleton identification methods, edge orientation methods, and causal effect estimation methods.

```python
from causal_pipe.pipe_config import (
    DataPreprocessingParams,
    CausalPipeConfig,
    VariableTypes,
    FASSkeletonMethod,
    FCIOrientationMethod,
    CausalEffectMethod,
)

# Define preprocessing parameters
preprocessor_params = DataPreprocessingParams(
    cat_to_codes=False,
    standardize=True,
    # keep_only_correlated_with=None,
    # filter_method="mi",
    # filter_threshold=0.1,
    handling_missing="impute",
    imputation_method="mice",
    use_r_mice=True,
    full_obs_cols=None,
)

# Define variable types
variable_types = VariableTypes(
    continuous=["age", "income"],
    ordinal=["education_level"],
    nominal=["gender", "diagnosis_1", "diagnosis_2"],
)

# Initialize the configuration
config = CausalPipeConfig(
    variable_types=variable_types,
    preprocessing_params=preprocessor_params,
    skeleton_method=FASSkeletonMethod(),
    orientation_method=FCIOrientationMethod(),
    causal_effect_methods=[CausalEffectMethod(name="pearson")],
    study_name="causal_analysis",
    output_path="./output",
    show_plots=True,
    verbose=True,
)

```

### 2. Initializing CausalPipe

Create an instance of the `CausalPipe` class by passing the configuration object.

```python
from causal_pipe import CausalPipe

# Initialize the toolkit
causal_pipe = CausalPipe(config)
```

### 3. Running the Causal Discovery Pipeline

Use the `run_pipeline` method to execute the full causal discovery process, including data preprocessing, skeleton identification, edge orientation, and causal effect estimation.

```python
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Run the causal discovery pipeline
causal_pipe.run_pipeline(data)
```

## Usage Examples

### Example: Running the Full Pipeline

Below is an example demonstrating how to configure and run the full causal discovery pipeline using `CausalPipe`.

```python
import numpy as np
import pandas as pd 

# Create a dummy DataFrame
np.random.seed(42)
df = pd.DataFrame(
    {
        "age": np.random.randint(20, 70, size=100),
        "income": np.random.normal(50000, 15000, size=100),
        "education_level": np.random.randint(1, 5, size=100),
        "gender": np.random.choice(["Male", "Female"], size=100),
        "diagnosis_1": np.random.randint(0, 2, size=100),
        "diagnosis_2": np.random.randint(0, 2, size=100),
    }
)

# Run the causal discovery pipeline
causal_pipe.run_pipeline(df)

# Access causal effects
print("Causal Effects:", causal_pipe.causal_effects)
```

### Example: Custom Configuration

Customize the skeleton identification and orientation methods to suit your specific analysis needs.

```python
# Define preprocessing parameters
preprocessor_params = DataPreprocessingParams(
    cat_to_codes=True,
    standardize=False,
    keep_only_correlated_with=None,
    filter_method="pearson",
    filter_threshold=0.2,
    handling_missing="drop",
    imputation_method="mice",
    use_r_mice=True,
    full_obs_cols=["age"],
)
 
# Initialize the configuration with BCSL skeleton method and Hill Climbing orientation
config = CausalPipeConfig(
    variable_types=variable_types,
    preprocessing_params=preprocessor_params,
    skeleton_method=BCSLSkeletonMethod(
        num_bootstrap_samples=200,
        multiple_comparison_correction="fdr",
        bootstrap_all_edges=True,
        use_aee_alpha=0.05,
        max_k=3,
    ),
    orientation_method=HillClimbingOrientationMethod(
        max_k=3,
        multiple_comparison_correction="fdr",
    ),
    causal_effect_methods=[
        CausalEffectMethod(name="sem"),
        CausalEffectMethod(name="pearson"),
    ],
    study_name="custom_causal_analysis",
    output_path="./output/custom_analysis",
    show_plots=True,
    verbose=True,
)

# Initialize the toolkit
causal_pipe = CausalPipe(config)

# Load your data
data = pd.read_csv("your_custom_data.csv")

# Run the causal discovery pipeline
causal_pipe.run_pipeline(data)

# Access causal effects
print("Causal Effects:", causal_pipe.causal_effects)
```

## Documentation

Comprehensive documentation is available to help you get started with CausalPipe and explore its full range of functionalities. Visit the [CausalPipe Documentation](https://albertbuchard.github.io/causal-pipe/) for detailed guides, API references, and tutorials.

## Contributing

Contributions are welcome! If you'd like to contribute to CausalPipe, please follow these steps:

1. **Fork the Repository:** Click the "Fork" button at the top-right corner of the repository page.
2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/your-username/causal-pipe.git
   ```
3. **Create a Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Commit Your Changes:**
   ```bash
   git commit -m "Add your detailed description here"
   ```
5. **Push to Your Fork:**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request:** Navigate to the original repository and click "Compare & pull request."

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, feel free to reach out:

- **Author:** Albert Buchard
- **Email:** [albert.buchard@gmail.com](mailto:albert.buchard@gmail.com)
- **GitHub:** [https://github.com/albertbuchard/causal-pipe](https://github.com/albertbuchard/causal-pipe)

---

### Additional Notes

- **Visualization Outputs:** Ensure that the output directory specified in the configuration exists or is created by CausalPipe. The toolkit will save visualizations like correlation graphs, skeletons, oriented graphs, and SEM results in the specified `output_path`.
  
- **R Package Dependencies:** Since CausalPipe integrates with R's `lavaan` and `mice` packages, make sure that R is installed on your system and that these packages are accessible. The toolkit attempts to install missing R packages automatically, but you may need to configure R's library paths or permissions accordingly.

- **Error Handling:** The toolkit includes error handling to catch and report issues during data preprocessing, model fitting, and causal effect estimation. Pay attention to console outputs for any warnings or error messages that may require your attention.

- **Extensibility:** CausalPipe is designed to be modular. You can extend its functionalities by adding new methods for skeleton identification, edge orientation, or causal effect estimation by creating new dataclasses and integrating them into the pipeline.

- **Performance Considerations:** Some methods, especially those involving multiple imputation or complex SEM models, can be computationally intensive. Ensure that your system has sufficient resources, and consider optimizing parameters like `num_bootstrap_samples` or `max_iter` based on your dataset's size and complexity.

By following this guide and leveraging the provided examples, you can effectively utilize **CausalPipe** to perform sophisticated causal discovery and analysis on your datasets.
 