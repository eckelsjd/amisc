"""Module with type hints for the AMISC package.

Includes:

- `MultiIndex` — tuples of integers or similar string representations
- `Dataset` — a type hint for the input/output `dicts` handled by `Component.model`
- `TrainIteration` — the results of a single training iteration
- `CompressionData` — a dictionary spec for passing data to/from `Variable.compress()`
- `LATENT_STR_ID` — a string identifier for latent coefficients of field quantities
"""
import ast as _ast
from pathlib import Path as _Path
from typing import Optional as _Optional

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike
from typing_extensions import TypedDict as _TypedDict

__all__ = ["MultiIndex", "Dataset", "TrainIteration", "CompressionData", "LATENT_STR_ID"]

LATENT_STR_ID = "_LATENT"  # String identifier for latent variables


class MultiIndex(tuple):
    """A multi-index is a tuple of integers, can be converted from a string."""
    def __new__(cls, __tuple=()):
        if isinstance(__tuple, str):
            return super().__new__(cls, map(int, _ast.literal_eval(__tuple)))
        else:
            return super().__new__(cls, map(int, __tuple))


class Dataset(_TypedDict, total=False):
    """Type hint for the input/output `dicts` of a call to `Component.model`. The keys are the variable names and the
    values are the corresponding `np.ndarrays`. There are also a few special keys that can be returned by the model
    that are described below.

    The model can return additional items that are not part of `Component.outputs`. These items are returned as object
    arrays in the output.

    This data structure is very similar to the `Dataset` class in the `xarray` package. Later versions might
    consider migrating to `xarray` for more advanced data manipulation.

    :ivar model_cost: the computational cost (seconds of CPU time) of a single model evaluation
    :ivar output_path: the path to the output file or directory written by the model
    :ivar errors: a `dict` with the indices where the model evaluation failed with context about the errors
    """
    model_cost: float | list | _ArrayLike
    output_path: str | _Path
    errors: dict


class TrainIteration(_TypedDict):
    """Gives the results of a single training iteration.

    :ivar component: the name of the component selected for refinement at this iteration
    :ivar alpha: the selected candidate model fidelity multi-index
    :ivar beta: the selected candidate surrogate fidelity multi-index
    :ivar num_evals: the number of model evaluations performed during this iteration
    :ivar added_cost: the total added computational cost of the new model evaluations
    :ivar added_error: the error/difference between the refined surrogate and the previous surrogate
    :ivar test_error: the error of the refined surrogate on the test set (optional)
    """
    component: str
    alpha: MultiIndex
    beta: MultiIndex
    num_evals: int
    added_cost: float
    added_error: float
    test_error: _Optional[dict[str, float]]


class CompressionData(_TypedDict, total=False):
    """Configuration `dict` for passing compression data to/from `Variable.compress()`.

    !!! Info "Field quantity shapes"
        Field quantity data can take on any arbitrary shape, which we indicate with `qty.shape`. For example, a 3d
        structured grid might have `qty.shape = (10, 15, 10)`. Unstructured data might just have `qty.shape = (N,)`
        for $N$ points in an unstructured grid. Regardless, `Variable.compress()` will flatten this and compress
        to a single latent vector of size `latent_size`. That is, `qty.shape` &rarr; `latent_size`.

    !!! Note "Compression coordinates"
        Field quantity data must be specified along with its coordinate locations. If the coordinate locations are
        different from what was used when building the compression map (i.e. the SVD data matrix), then they will be
        interpolated to/from the SVD coordinates.

    :ivar coords: `(qty.shape, dim)` the coordinate locations of the qty data; coordinates exist in `dim` space (e.g.
                 `dim=2` for 2d Cartesian coordinates). Defaults to the coordinates used when building the construction
                  map (i.e. the coordinates of the data in the SVD data matrix)
    :ivar latent: `(..., latent_size)` array of latent space coefficients for a field quantity; this is what is
                  _returned_ by `Variable.compress()` and what is _expected_ as input by `Variable.reconstruct()`.
    :ivar qty: `(..., qty.shape)` array of uncompressed field quantity data for this qty within
               the `fields` list. Each qty in this list will be its own `key:value` pair in the
               `CompressionData` structure
    """
    coords: _np.ndarray
    latent: _np.ndarray
    qty: _np.ndarray
