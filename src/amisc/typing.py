"""Module with type hints for the AMISC package.

Includes:

- `MultiIndex` — tuples of integers or similar string representations
- `Dataset` — a type hint for the input/output `dicts` handled by `Component.model`
- `builtin` — common built-in Python objects
- `TrainIteration` — the results of a single training iteration
- `CompressionData` — a dictionary spec for passing data to/from `Variable.compress()`
"""
from pathlib import Path as _Path
from typing import Optional as _Optional

from typing_extensions import TypedDict as _TypedDict

import numpy as _np
from numpy.typing import ArrayLike as _ArrayLike


__all__ = ["MultiIndex", "Dataset", "builtin", "TrainIteration", "CompressionData"]

MultiIndex = str | tuple[int, ...]  # A multi-index is a tuple of integers or a similar string representation
builtin = str | dict | list | int | float | tuple | bool  # Generic type for common built-in Python objects


class Dataset(_TypedDict, total=False):
    """Type hint for the input/output `dicts` of a call to `Component.model`. The keys are the variable names and the
    values are the corresponding np arrays. There are also a few special keys that can be returned by the model:

    - `model_cost` — the computational cost (seconds of CPU time) of a single model evaluation
    - `output_path` — the path to the output file or directory written by the model
    - `errors` — a `dict` with the indices where the model evaluation failed with context about the errors

    The model can return additional items that are not part of `Component.outputs`. These items are returned as object
    arrays in the output.

    This data structure is very similar to the `Dataset` class in the `xarray` package. Later versions might
    consider migrating to `xarray` for more advanced data manipulation.
    """
    model_cost: float | list | _ArrayLike
    output_path: str | _Path
    errors: dict


class TrainIteration(_TypedDict):
    """Gives the results of a single training iteration."""
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

    !!! Info "Compression coordinates"
        Field quantity data must be specified along with its coordinate locations. If the coordinate locations are
        different from what was used when building the compression map (i.e. the SVD data matrix), then they will be
        interpolated to/from the SVD coordinates.

    :ivar coord: `(qty.shape, dim)` the coordinate locations of the qty data; coordinates exist in `dim` space (e.g.
                 `dim=2` for 2d Cartesian coordinates). Defaults to the coordinates used when building the construction
                  map (i.e. the coordinates of the data in the SVD data matrix)
    :ivar latent: `(..., latent_size)` array of latent space coefficients for a field quantity; this is what is
                  _returned_ by `Variable.compress()` and what is _expected_ as input by `Variable.reconstruct()`.
    :ivar qty: `(..., qty.shape)` array of uncompressed field quantity data for this qty within
               the `fields` list. Each qty in this list will be its own `key:value` pair in the
               `CompressionData` structure
    """
    coord: _np.ndarray
    latent: _np.ndarray
    qty: _np.ndarray
