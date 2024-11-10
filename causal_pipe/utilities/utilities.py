import json
import random
import warnings
from typing import Optional

import numpy as np


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
