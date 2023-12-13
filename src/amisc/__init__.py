"""Adaptive multi-index stochastic collocation for metamodeling/surrogates of multidisciplinary systems.

- Author - Joshua Eckels (eckelsjd@umich.edu)
- License - GNU GPLv3
"""
import numpy as np

from amisc.interpolator import BaseInterpolator
from amisc.rv import BaseRV

__version__ = "0.1.0"

# Custom types that are used frequently
IndexSet = list[tuple[tuple, tuple]]
MiscTree = dict[str: dict[str: float | BaseInterpolator]]
InterpResults = BaseInterpolator | tuple[list[int | tuple | str], np.ndarray, BaseInterpolator]
IndicesRV = list[int | str | BaseRV] | int | str | BaseRV
