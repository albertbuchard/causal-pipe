from dataclasses import dataclass
from typing import Dict, Callable, Optional

import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph


@dataclass
class PySRFitterOutput:
    structural_equations: Dict[str, Dict]
    final_graph: GeneralGraph


PySRFitterType = Callable[[pd.DataFrame, GeneralGraph, Optional[Dict]], PySRFitterOutput]
