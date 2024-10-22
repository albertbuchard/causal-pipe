
# Installation

Follow these instructions to install **CausalPipe** on your system.

### Requirements
- Python 3.6 or higher
- R (required for `lavaan` and `mice` integration)

### Installing via PyPI
You can install **CausalPipe** using `pip`:

```bash
pip install causal-pipe
```

### Additional Dependencies
CausalPipe relies on several Python and R packages. Ensure that the following dependencies are installed:

#### Python Packages:
```bash
numpy>=1.18.0
scipy>=1.4.0
scikit-learn>=0.22.0
causal-learn==0.1.3.8
bcsl-python==0.8.0
rpy2==3.5.16
npeet-plus==0.2.0
networkx==3.2.1
pandas==2.2.3
factor_analyzer==0.5.1
```

#### R Packages:
- `lavaan` (for Structural Equation Modeling)
- `mice` (for multiple imputation)

Ensure that R is properly installed on your system and the necessary packages are available.
