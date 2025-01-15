"""Classes for storing and managing training data for surrogate models. The `TrainingData` interface also
specifies how new training data should be sampled over the input space (i.e. experimental design).

Includes:

- `TrainingData` — an interface for storing surrogate training data.
- `SparseGrid` — a class for storing training data in a sparse grid format.
"""
from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import direct

from amisc.serialize import PickleSerializable, Serializable
from amisc.typing import LATENT_STR_ID, Dataset, MultiIndex
from amisc.utils import _RidgeRegression

__all__ = ['TrainingData', 'SparseGrid']


class TrainingData(Serializable, ABC):
    """Interface for storing and collecting surrogate training data. `TrainingData` objects should:

    - `get` - retrieve the training data
    - `set` - store the training data
    - `refine` - generate new design points for the parent `Component` model
    - `clear` - clear all training data
    - `set_errors` - store error information (if desired)
    - `impute_missing_data` - fill in missing values in the training data (if desired)
    """

    @abstractmethod
    def get(self, alpha: MultiIndex, beta: MultiIndex, y_vars: list[str] = None,
            skip_nan: bool = False) -> tuple[Dataset, Dataset]:
        """Return the training data for a given multi-index pair.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param y_vars: the keys of the outputs to return (if `None`, return all outputs)
        :param skip_nan: skip any data points with remaining `nan` values if `skip_nan=True`
        :returns: `dicts` of model inputs `x_train` and outputs `y_train`
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, alpha: MultiIndex, beta: MultiIndex, coords: list[Any], yi_dict: Dataset):
        """Store training data for a given multi-index pair.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param coords: locations for storing the `yi` values in the underlying data structure
        :param yi_dict: a `dict` of model output `yi` values, each entry should be the same length as `coords`
        """
        raise NotImplementedError

    @abstractmethod
    def set_errors(self, alpha: MultiIndex, beta: MultiIndex, coords: list[Any], errors: list[dict]):
        """Store error information for a given multi-index pair (just pass if you don't care).

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param coords: locations for storing the error information in the underlying data structure
        :param errors: a list of error dictionaries, should be the same length as `coords`
        """
        raise NotImplementedError

    @abstractmethod
    def impute_missing_data(self, alpha: MultiIndex, beta: MultiIndex):
        """Impute missing values in the training data for a given multi-index pair (just pass if you don't care).

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        """
        raise NotImplementedError

    @abstractmethod
    def refine(self, alpha: MultiIndex, beta: MultiIndex, input_domains: dict[str, tuple],
               weight_fcns: dict[str, callable] = None) -> tuple[list[Any], Dataset]:
        """Return new design/training points for a given multi-index pair and their coordinates/locations in the
        `TrainingData` storage structure.

        !!! Example
            ```python
            domains = {'x1': (0, 1), 'x2': (0, 1)}
            alpha, beta = (0, 1), (1, 1)
            coords, x_train = training_data.refine(alpha, beta, domains)
            y_train = my_model(x_train)
            training_data.set(alpha, beta, coords, y_train)
            ```

        The returned data coordinates `coords` should be any object that can be used to locate the corresponding
        `x_train` training points in the `TrainingData` storage structure. These `coords` will be passed back to the
        `set` function to store the training data at a later time (i.e. after model evaluation).

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param input_domains: a `dict` specifying domain bounds for each input variable
        :param weight_fcns: a `dict` of weighting functions for each input variable
        :returns: a list of new data coordinates `coords` and the corresponding training points `x_train`
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        """Clear all training data."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, config: dict) -> TrainingData:
        """Create a `TrainingData` object from a `dict` configuration. Currently, only `method='sparse-grid'` is
        supported for the `SparseGrid` class.
        """
        method = config.pop('method', 'sparse-grid').lower()
        match method:
            case 'sparse-grid':
                return SparseGrid(**config)
            case other:
                raise NotImplementedError(f"Unknown training data method: {other}")


@dataclass
class SparseGrid(TrainingData, PickleSerializable):
    """A class for storing training data in a sparse grid format. The `SparseGrid` class stores training points
    by their coordinate location in a larger tensor-product grid, and obtains new training data by refining
    a single 1d grid at a time.

    !!! Note "MISC and sparse grids"
        MISC itself can be thought of as an extension to the well-known sparse grid technique, so this class
        readily integrates with the MISC implementation in `Component`. Sparse grids limit the curse
        of dimensionality up to about `dim = 10-15` for the input space (which would otherwise be infeasible with a
        normal full tensor-product grid of the same size).

    !!! Info "About points in a sparse grid"
        A sparse grid approximates a full tensor-product grid $(N_1, N_2, ..., N_d)$, where $N_i$ is the number of grid
        points along dimension $i$, for a $d$-dimensional space. Each point is uniquely identified in the sparse grid
        by a list of indices $(j_1, j_2, ..., j_d)$, where $j_i = 0 ... N_i$. We refer to this unique identifier as a
        "grid coordinate". In the `SparseGrid` data structure, these coordinates are used along with the `alpha`
        fidelity index to uniquely locate the training data for a given multi-index pair.

    :ivar collocation_rule: the collocation rule to use for generating new grid points (only 'leja' is supported)
    :ivar knots_per_level: the number of grid knots/points per level in the `beta` fidelity multi-index
    :ivar expand_latent_method: method for expanding latent grids, either 'round-robin' or 'tensor-product'
    :ivar opt_args: extra arguments for the global 1d `direct` optimizer
    :ivar betas: a set of all `beta` multi-indices that have been seen so far
    :ivar x_grids: a `dict` of grid points for each 1d input dimension
    :ivar yi_map: a `dict` of model outputs for each grid coordinate
    :ivar yi_nan_map: a `dict` of imputed model outputs for each grid coordinate where the model failed (or gave nan)
    :ivar error_map: a `dict` of error information for each grid coordinate where the model failed
    :ivar latent_size: the number of latent coefficients for each variable (0 if scalar)
    """
    MAX_IMPUTE_SIZE: ClassVar[int] = 10        # don't try to impute large arrays

    collocation_rule: str = 'leja'
    knots_per_level: int = 2
    expand_latent_method: str = 'round-robin'  # or 'tensor-product', for converting beta to latent grid sizes
    opt_args: dict = field(default_factory=lambda: {'locally_biased': False, 'maxfun': 300})  # for leja optimizer

    betas: set[MultiIndex] = field(default_factory=set)
    x_grids: dict[str, ArrayLike] = field(default_factory=dict)
    yi_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, ArrayLike]]] = field(default_factory=dict)
    yi_nan_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, ArrayLike]]] = field(default_factory=dict)
    error_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, Any]]] = field(default_factory=dict)
    latent_size: dict[str, int] = field(default_factory=dict)  # keep track of latent grid sizes for each variable

    def clear(self):
        """Clear all training data."""
        self.betas.clear()
        self.x_grids.clear()
        self.yi_map.clear()
        self.yi_nan_map.clear()
        self.error_map.clear()
        self.latent_size.clear()

    def get_by_coord(self, alpha: MultiIndex, coords: list, y_vars: list = None, skip_nan: bool = False):
        """Get training data from the sparse grid for a given `alpha` and list of grid coordinates. Try to replace
        `nan` values with imputed values. Skip any data points with remaining `nan` values if `skip_nan=True`.

        :param alpha: the model fidelity indices
        :param coords: a list of grid coordinates to locate the `yi` values in the sparse grid data structure
        :param y_vars: the keys of the outputs to return (if `None`, return all outputs)
        :param skip_nan: skip any data points with remaining `nan` values if `skip_nan=True` (only for numeric outputs)
        :returns: `dicts` of model inputs `xi_dict` and outputs `yi_dict`
        """
        N = len(coords)
        is_numeric = {}
        is_singleton = {}
        xi_dict = self._extract_grid_points(coords)
        yi_dict = {}

        first_yi = next(iter(self.yi_map[alpha].values()))
        if y_vars is None:
            y_vars = first_yi.keys()

        for var in y_vars:
            yi = np.atleast_1d(first_yi[var])
            is_numeric[var] = self._is_numeric(yi)
            is_singleton[var] = self._is_singleton(yi)
            yi_dict[var] = np.empty(N, dtype=np.float64 if is_numeric[var] and is_singleton[var] else object)

        for i, coord in enumerate(coords):
            try:
                yi_curr = self.yi_map[alpha][coord]
                for var in y_vars:
                    yi = arr if (arr := self.yi_nan_map[alpha].get(coord, {}).get(var)) is not None else yi_curr[var]
                    yi_dict[var][i] = yi if is_singleton[var] else np.atleast_1d(yi)

            except KeyError as e:
                raise KeyError(f"Can't access sparse grid data for alpha={alpha}, coord={coord}. "
                               f"Make sure the data has been set first.") from e

        # Delete nans if requested (only for numeric singleton outputs)
        if skip_nan:
            nan_idx = np.full(N, False)
            for var in y_vars:
                if is_numeric[var] and is_singleton[var]:
                    nan_idx |= np.isnan(yi_dict[var])

            xi_dict = {k: v[~nan_idx] for k, v in xi_dict.items()}
            yi_dict = {k: v[~nan_idx] for k, v in yi_dict.items()}

        return xi_dict, yi_dict  # Both with elements of shape (N, ...) for N grid points

    def get(self, alpha: MultiIndex, beta: MultiIndex, y_vars: list[str] = None, skip_nan: bool = False):
        """Get the training data from the sparse grid for a given `alpha` and `beta` pair."""
        return self.get_by_coord(alpha, list(self._expand_grid_coords(beta)), y_vars=y_vars, skip_nan=skip_nan)

    def set_errors(self, alpha: MultiIndex, beta: MultiIndex, coords: list, errors: list[dict]):
        """Store error information in the sparse-grid for a given multi-index pair."""
        for coord, error in zip(coords, errors):
            self.error_map[alpha][coord] = copy.deepcopy(error)

    def set(self, alpha: MultiIndex, beta: MultiIndex, coords: list, yi_dict: dict[str, ArrayLike]):
        """Store model output `yi_dict` values.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param coords: a list of grid coordinates to locate the `yi` values in the sparse grid data structure
        :param yi_dict: a `dict` of model output `yi` values
        """
        for i, coord in enumerate(coords):  # First dim of yi is loop dim aligning with coords
            new_yi = {}
            for var, yi in yi_dict.items():
                yi = np.atleast_1d(yi[i])
                new_yi[var] = (float(yi[0]) if self._is_numeric(yi) else yi[0]) if self._is_singleton(yi) else yi.tolist()  # noqa: E501
            self.yi_map[alpha][coord] = copy.deepcopy(new_yi)

    def impute_missing_data(self, alpha: MultiIndex, beta: MultiIndex):
        """Impute missing values in the sparse grid for a given multi-index pair by linear regression imputation."""
        imputer, xi_all, yi_all = None, None, None

        # only impute (small-length) numeric quantities
        yi_dict = next(iter(self.yi_map[alpha].values()))
        output_vars = [var for var in self._numeric_outputs(yi_dict)
                       if len(np.ravel(yi_dict[var])) <= self.MAX_IMPUTE_SIZE]

        for coord, yi_dict in self.yi_map[alpha].items():
            if any([np.any(np.isnan(yi_dict[var])) for var in output_vars]):
                if imputer is None:
                    # Grab all 'good' interpolation points and train a simple linear regression fit
                    xi_all, yi_all = self.get(alpha, beta, y_vars=output_vars, skip_nan=True)
                    if len(xi_all) == 0 or len(next(iter(xi_all.values()))) == 0:
                        continue  # possible if no good data has been set yet

                    N = next(iter(xi_all.values())).shape[0]  # number of grid points
                    xi_mat = np.concatenate([xi_all[var][:, np.newaxis] if len(xi_all[var].shape) == 1 else
                                             xi_all[var] for var in xi_all.keys()], axis=-1)
                    yi_mat = np.concatenate([yi_all[var][:, np.newaxis] if len(yi_all[var].shape) == 1 else
                                             yi_all[var].reshape((N, -1)) for var in output_vars], axis=-1)

                    imputer = _RidgeRegression(alpha=1.0)
                    imputer.fit(xi_mat, yi_mat)

                # Run the imputer for this coordinate
                x_interp = self._extract_grid_points(coord)
                x_interp = np.concatenate([x_interp[var][:, np.newaxis] if len(x_interp[var].shape) == 1 else
                                           x_interp[var] for var in x_interp.keys()], axis=-1)
                y_interp = imputer.predict(x_interp)

                # Unpack the imputed value
                y_impute = {}
                start_idx = 0
                for var in output_vars:
                    var_shape = yi_all[var].shape[1:] or (1,)
                    end_idx = start_idx + int(np.prod(var_shape))
                    yi = np.atleast_1d(y_interp[0, start_idx:end_idx]).reshape(var_shape)
                    nan_idx = np.isnan(np.atleast_1d(yi_dict[var]))
                    yi[~nan_idx] = np.atleast_1d(yi_dict[var])[~nan_idx]  # Only keep imputed values where yi is nan
                    y_impute[var] = float(yi[0]) if self._is_singleton(yi) else yi.tolist()
                    start_idx = end_idx

                self.yi_nan_map[alpha][coord] = copy.deepcopy(y_impute)

    def refine(self, alpha: MultiIndex, beta: MultiIndex, input_domains: dict, weight_fcns: dict = None):
        """Refine the sparse grid for a given `alpha` and `beta` pair and given collocation rules. Return any new
        grid points that do not have model evaluations saved yet.

        !!! Note
            The `beta` multi-index is used to determine the number of collocation points in each input dimension. The
            length of `beta` should therefore match the number of variables in `x_vars`.
        """
        weight_fcns = weight_fcns or {}

        # Initialize a sparse grid for beta=(0, 0, ..., 0)
        if np.sum(beta) == 0:
            if len(self.x_grids) == 0:
                num_latent = {}
                for var in input_domains:
                    if LATENT_STR_ID in var:
                        base_id = var.split(LATENT_STR_ID)[0]
                        num_latent[base_id] = 1 if base_id not in num_latent else num_latent[base_id] + 1
                    else:
                        num_latent[var] = 0
                self.latent_size = num_latent

                new_pt = {}
                domains = iter(input_domains.items())
                for grid_size in self.beta_to_knots(beta):
                    if isinstance(grid_size, int):  # scalars
                        var, domain = next(domains)
                        new_pt[var] = self.collocation_1d(grid_size, domain, method=self.collocation_rule,
                                                          wt_fcn=weight_fcns.get(var, None),
                                                          opt_args=self.opt_args).tolist()
                    else:                           # latent coeffs
                        for s in grid_size:
                            var, domain = next(domains)
                            new_pt[var] = self.collocation_1d(s, domain, method=self.collocation_rule,
                                                              wt_fcn=weight_fcns.get(var, None),
                                                              opt_args=self.opt_args).tolist()
                self.x_grids = new_pt
            self.betas.add(beta)
            self.yi_map.setdefault(alpha, dict())
            self.yi_nan_map.setdefault(alpha, dict())
            self.error_map.setdefault(alpha, dict())
            new_coords = list(self._expand_grid_coords(beta))
            return new_coords, self._extract_grid_points(new_coords)

        # Otherwise, refine the sparse grid
        for beta_old in self.betas:
            # Get the first lower neighbor in the sparse grid and refine the 1d grid if necessary
            if self.is_one_level_refinement(beta_old, beta):
                new_grid_size = self.beta_to_knots(beta)
                inputs = zip(self.x_grids.keys(), self.x_grids.values(), input_domains.values())

                for new_size in new_grid_size:
                    if isinstance(new_size, int):       # scalar grid
                        var, grid, domain = next(inputs)
                        if len(grid) < new_size:
                            num_new_pts = new_size - len(grid)
                            self.x_grids[var] = self.collocation_1d(num_new_pts, domain, grid, opt_args=self.opt_args,
                                                                    wt_fcn=weight_fcns.get(var, None),
                                                                    method=self.collocation_rule).tolist()
                    else:                               # latent grid
                        for s_new in new_size:
                            var, grid, domain = next(inputs)
                            if len(grid) < s_new:
                                num_new_pts = s_new - len(grid)
                                self.x_grids[var] = self.collocation_1d(num_new_pts, domain, grid,
                                                                        opt_args=self.opt_args,
                                                                        wt_fcn=weight_fcns.get(var, None),
                                                                        method=self.collocation_rule).tolist()
                break

        new_coords = []
        for coord in self._expand_grid_coords(beta):
            if coord not in self.yi_map[alpha]:
                # If we have not computed this grid coordinate yet
                new_coords.append(coord)

        new_pts = self._extract_grid_points(new_coords)

        self.betas.add(beta)
        return new_coords, new_pts

    def _extract_grid_points(self, coords: list[tuple] | tuple):
        """Extract the `x` grid points located at `coords` from `x_grids` and return as the `pts` dictionary."""
        if not isinstance(coords, list):
            coords = [coords]
        pts = {var: np.empty(len(coords)) for var in self.x_grids}

        for k, coord in enumerate(coords):
            grids = iter(self.x_grids.items())
            for idx in coord:
                if isinstance(idx, int):        # scalar grid point
                    var, grid = next(grids)
                    pts[var][k] = grid[idx]
                else:                           # latent coefficients
                    for i in idx:
                        var, grid = next(grids)
                        pts[var][k] = grid[i]

        return pts

    def _expand_grid_coords(self, beta: MultiIndex):
        """Iterable over all grid coordinates for a given `beta`, accounting for scalars and latent coefficients."""
        grid_size = self.beta_to_knots(beta)
        grid_coords = []
        for s in grid_size:
            if isinstance(s, int):              # scalars
                grid_coords.append(range(s))
            else:                               # latent coefficients
                grid_coords.append(itertools.product(*[range(latent_size) for latent_size in s]))

        yield from itertools.product(*grid_coords)

    @staticmethod
    def _is_singleton(arr: np.ndarray):
        return len(arr.shape) == 1 and arr.shape[0] == 1

    @staticmethod
    def _is_numeric(arr: np.ndarray):
        return np.issubdtype(arr.dtype, np.number)

    @classmethod
    def _numeric_outputs(cls, yi_dict: dict[str, ArrayLike]) -> list[str]:
        """Return a list of the output variables that have numeric data."""
        output_vars = []
        for var in yi_dict.keys():
            try:
                if cls._is_numeric(np.atleast_1d(yi_dict[var])):
                    output_vars.append(var)
            except Exception:
                continue
        return output_vars

    @staticmethod
    def is_one_level_refinement(beta_old: tuple, beta_new: tuple) -> bool:
        """Check if a new `beta` multi-index is a one-level refinement from a previous `beta`.

        !!! Example
            Refining from `(0, 1, 2)` to the new multi-index `(1, 1, 2)` is a one-level refinement. But refining to
            either `(2, 1, 2)` or `(1, 2, 2)` are not, since more than one refinement occurs at the same time.

        :param beta_old: the starting multi-index
        :param beta_new: the new refined multi-index
        :returns: whether `beta_new` is a one-level refinement from `beta_old`
        """
        level_diff = np.array(beta_new, dtype=int) - np.array(beta_old, dtype=int)
        ind = np.nonzero(level_diff)[0]
        return ind.shape[0] == 1 and level_diff[ind] == 1

    def beta_to_knots(self, beta: MultiIndex, knots_per_level: int = None, latent_size: dict = None,
                      expand_latent_method: str = None) -> tuple:
        """Convert a `beta` multi-index to the number of knots per dimension in the sparse grid.

        :param beta: refinement level indices
        :param knots_per_level: level-to-grid-size multiplier, i.e. number of new points (or knots) for each beta level
        :param latent_size: the number of latent coefficients for each variable (0 if scalar); number of variables and
                            order should match the `beta` multi-index
        :param expand_latent_method: method for expanding latent grids, either 'round-robin' or 'tensor-product'
        :returns: the number of knots/points per dimension for the sparse grid
        """
        knots_per_level = knots_per_level or self.knots_per_level
        latent_size = latent_size or self.latent_size
        expand_latent_method = expand_latent_method or self.expand_latent_method

        grid_size = []
        for i, (var, num_latent) in enumerate(latent_size.items()):
            if num_latent > 0:
                match expand_latent_method:
                    case 'round-robin':
                        if beta[i] == 0:
                            grid_size.append((1,) * num_latent)  # initializes all latent grids to 1
                        else:
                            latent_refine_idx = (beta[i] - 1) % num_latent
                            latent_refine_num = ((beta[i] - 1) // num_latent) + 1
                            latent_beta = tuple([latent_refine_num] * (latent_refine_idx + 1) +
                                                [latent_refine_num - 1] * (num_latent - latent_refine_idx - 1))
                            latent_grid = [knots_per_level * latent_beta[j] + 1 for j in range(num_latent)]
                            grid_size.append(tuple(latent_grid))
                    case 'tensor-product':
                        grid_size.append((knots_per_level * beta[i] + 1,) * num_latent)
                    case other:
                        raise NotImplementedError(f"Unknown method for expanding latent grids: {other}")
            else:
                grid_size.append(knots_per_level * beta[i] + 1)

        return tuple(grid_size)

    @staticmethod
    def collocation_1d(N: int, z_bds: tuple, z_pts: np.ndarray = None,
                       wt_fcn: callable = None, method='leja', opt_args=None) -> np.ndarray:
        """Find the next `N` points in the 1d sequence of `z_pts` using the provided collocation method.

        :param N: number of new points to add to the sequence
        :param z_bds: bounds on the 1d domain
        :param z_pts: current univariate sequence `(Nz,)`, start at middle of `z_bds` if `None`
        :param wt_fcn: weighting function, uses a constant weight if `None`, callable as `wt_fcn(z)`
        :param method: collocation method to use, currently only 'leja' is supported
        :param opt_args: extra arguments for the global 1d `direct` optimizer
        :returns: the univariate sequence `z_pts` augmented by `N` new points
        """
        opt_args = opt_args or {}
        if wt_fcn is None:
            wt_fcn = lambda z: 1
        if z_pts is None:
            z_pts = (z_bds[1] + z_bds[0]) / 2
            N = N - 1
        z_pts = np.atleast_1d(z_pts)

        match method:
            case 'leja':
                # Construct Leja sequence by maximizing the Leja objective sequentially
                for i in range(N):
                    obj_fun = lambda z: -wt_fcn(np.array(z)) * np.prod(np.abs(z - z_pts))
                    res = direct(obj_fun, [z_bds], **opt_args)  # Use global DIRECT optimization over 1d domain
                    z_star = res.x
                    z_pts = np.concatenate((z_pts, z_star))
            case other:
                raise NotImplementedError(f"Unknown collocation method: {other}")

        return z_pts
