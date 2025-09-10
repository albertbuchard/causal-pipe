import json
import random
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd


def dump_json_to(data, path: str):
    """
    Dump data to a JSON file.

    Parameters:
    - data: Data to be dumped.
    - path (str): File path where the JSON will be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def set_seed_python_and_r(seed: Optional[int] = None):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (Optional[int]): Random seed value.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        try:
            import rpy2.robjects as robjects

            # Set the seed in R
            robjects.r(f"set.seed({seed})")
        except ImportError:
            warnings.warn("rpy2 not installed. R seed not set.")
        except Exception as e:
            warnings.warn(f"Error setting R seed: {str(e)}")

        print(f"Random seed set to {seed}.")
    else:
        print("Random seed is None - using default random initialization.")


def nodes_names_from_data(data: pd.DataFrame) -> List[str]:
    """
    Extracts node names from the dataset.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset from which to extract node names.

    Returns
    -------
    List[str]
        A list of variable names.

    Raises
    ------
    ValueError
        If the data type is unsupported.
    """
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    elif isinstance(data, np.ndarray):
        return [f"Var{i}" for i in range(data.shape[1])]
    else:
        raise ValueError(
            "Unsupported data type. Please provide a pandas DataFrame or numpy array."
        )
