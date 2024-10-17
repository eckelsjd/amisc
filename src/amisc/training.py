"""Classes for storing and managing training data for surrogate models.

Includes:

- `TrainingData` — an interface for storing surrogate training data.
- `SparseGrid` — a class for storing training data in a sparse grid format.
"""
from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import direct
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from numpy.typing import ArrayLike

from amisc import VariableList
from amisc.serialize import PickleSerializable, Serializable
from amisc.typing import MultiIndex, Dataset

__all__ = ['TrainingData', 'SparseGrid']


class TrainingData(Serializable, ABC):
    """Interface for storing surrogate training data."""

    @abstractmethod
    def get(self, alpha: MultiIndex, beta: MultiIndex, y_vars: list[str] = None) -> tuple[Dataset, Dataset]:
        """Return the training data for a given multi-index pair.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param y_vars: the keys of the outputs to return (if `None`, return all outputs)
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
    def refine(self, alpha: MultiIndex, beta: MultiIndex, x_vars: VariableList) -> tuple[list[Any], Dataset]:
        """Return new design/training points for a given multi-index pair.

        !!! Example
            ```python
            x_vars = VariableList(['x1', 'x2', 'x3'])
            alpha, beta = (0, 1), (1, 1, 2)
            coords, x_train = training_data.refine(alpha, beta, x_vars)
            y_train = my_model(x_train)
            training_data.set(alpha, beta, coords, y_train)
            ```

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param x_vars: the list of input variables
        :returns: a list of new grid coordinates `coords` and the corresponding training points `x_train`
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, config: dict) -> TrainingData:
        """Create a `TrainingData` object from a `dict` configuration."""
        method = config.pop('method', 'sparse-grid').lower()
        match method:
            case 'sparse-grid':
                return SparseGrid(**config)
            case other:
                raise NotImplementedError(f"Unknown training data method: {method}")


@dataclass
class SparseGrid(TrainingData, PickleSerializable):
    MAX_IMPUTE_SIZE: ClassVar[int] = 10        # don't try to impute large arrays

    collocation_rule: str = 'leja'
    knots_per_level: int = 2
    expand_latent_method: str = 'round-robin'  # or 'tensor-product', for converting beta to latent grid sizes

    betas: set[MultiIndex] = field(default_factory=set)
    x_grids: dict[str, ArrayLike] = field(default_factory=dict)
    yi_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, ArrayLike]]] = field(default_factory=dict)
    yi_nan_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, ArrayLike]]] = field(default_factory=dict)
    error_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, Any]]] = field(default_factory=dict)
    latent_size: dict[str, int] = field(default_factory=dict)  # keep track of latent grid sizes for each variable

    def get_by_coord(self, alpha: MultiIndex, coords, y_vars=None, skip_nan: bool = False):
        """Get training data from the sparse grid for a given `alpha` and list of grid coordinates. Try to replace
        `nan` values with imputed values. Skip any data points with remaining `nan` values if `skip_nan=True`.
        """
        xi_dict = {}
        yi_dict = {}
        y_squeeze = {}  # Keep track of whether to squeeze extra output dimensions (i.e. non field quantities)
        for coord in coords:
            try:
                skip_coord = False
                yi_curr = copy.deepcopy(self.yi_map[alpha][coord])
                output_vars = self._numeric_outputs(yi_curr)
                for var in output_vars:
                    if np.any(np.isnan(yi_curr[var])):
                        yi_curr[var] = self.yi_nan_map[alpha].get(coord, yi_curr)[var]
                    if skip_nan and np.any(np.isnan(yi_curr[var])):
                        skip_coord = True
                        break

                if not skip_coord:
                    self._append_grid_points(coord, xi_dict)
                    yi_vars = y_vars if y_vars is not None else yi_curr.keys()
                    for var in yi_vars:
                        yi = np.atleast_1d(yi_curr[var])
                        y_squeeze[var] = self._is_singleton(yi)  # squeeze for scalar quantities
                        yi = np.expand_dims(yi, axis=0)
                        yi_dict[var] = yi if yi_dict.get(var) is None else np.concatenate((yi_dict[var], yi), axis=0)
            except KeyError as e:
                raise ValueError(f"Can't access sparse grid data for alpha={alpha}, coord={coord}. "
                                 f"Make sure the data has been set first.") from e

        # Squeeze out extra dimension for scalar quantities
        for var in yi_dict.keys():
            if y_squeeze[var]:
                yi_dict[var] = np.squeeze(yi_dict[var], axis=-1)

        return xi_dict, yi_dict  # Both with elements of shape (N, ...) for N grid points

    def get(self, alpha: MultiIndex, beta: MultiIndex, y_vars: list[str] = None, skip_nan: bool = False):
        """Get the training data from the sparse grid for a given `alpha` and `beta` pair."""
        return self.get_by_coord(alpha, self._expand_grid_coords(beta), y_vars=y_vars, skip_nan=skip_nan)

    def set_errors(self, alpha: MultiIndex, beta: MultiIndex, coords: list, errors: list[dict]):
        """Store error information in the sparse-grid for a given multi-index pair."""
        for coord, error in zip(coords, errors):
            self.error_map[alpha][coord] = copy.deepcopy(error)

    def set(self, alpha: MultiIndex, beta: MultiIndex, coords: list, yi_dict: dict[str, ArrayLike]):
        """Store model output `yi` values.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param coords: a list of grid coordinates to locate the `yi` values in the sparse grid data structure
        :param yi_dict: a `dict` of model output `yi` values
        """
        yi_dict = copy.deepcopy(yi_dict)
        for i, coord in enumerate(coords):  # First dim of yi is loop dim aligning with coords
            new_yi = {}
            for var, yi in yi_dict.items():
                yi = np.atleast_1d(yi[i])
                new_yi[var] = (float(yi[0]) if self._is_numeric(yi) else yi[0]) if self._is_singleton(yi) else yi.tolist()
            self.yi_map[alpha][coord] = copy.deepcopy(new_yi)

    def impute_missing_data(self, alpha: MultiIndex, beta: MultiIndex):
        """Impute missing values in the sparse grid for a given multi-index pair by linear regression imputation."""
        imputer, xi_all, yi_all = None, None, None
        for coord, yi_dict in self.yi_map[alpha].items():
            # only impute (small-length) numeric quantities
            output_vars = [var for var in self._numeric_outputs(yi_dict)
                           if len(np.ravel(yi_dict[var])) <= self.MAX_IMPUTE_SIZE]

            if any([np.any(np.isnan(yi_dict[var])) for var in output_vars]):
                if imputer is None:
                    # Grab all 'good' interpolation points and train a simple linear regression fit
                    xi_all, yi_all = self.get(alpha, beta, y_vars=output_vars, skip_nan=True)
                    if len(xi_all) == 0:
                        continue  # possible if no good data has been set yet

                    N = next(iter(xi_all.values())).shape[0]  # number of grid points
                    xi_mat = np.concatenate([xi_all[var][:, np.newaxis] if len(xi_all[var].shape) == 1 else
                                             xi_all[var] for var in xi_all.keys()], axis=-1)
                    yi_mat = np.concatenate([yi_all[var][:, np.newaxis] if len(yi_all[var].shape) == 1 else
                                             yi_all[var].reshape((N, -1)) for var in output_vars], axis=-1)

                    imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
                    imputer.fit(xi_mat, yi_mat)

                # Run the imputer for this coordinate
                x_interp = self._append_grid_points(coord)
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
                    y_impute[var] = float(yi) if self._is_singleton(yi) else yi.tolist()
                    start_idx = end_idx

                self.yi_nan_map[alpha][coord] = copy.deepcopy(y_impute)

    def refine(self, alpha: MultiIndex, beta: MultiIndex, x_vars: VariableList):
        """Refine the sparse grid for a given `alpha` and `beta` pair and given collocation rules. Return any new
        grid points that do not have model evaluations saved yet.

        !!! Note
            The `beta` multi-index is used to determine the number of collocation points in each input dimension. The
            length of `beta` should therefore match the number of variables in `x_vars`.
        """
        # Initialize a sparse grid for beta=(0, 0, ..., 0)
        if np.sum(beta) == 0:
            if len(self.x_grids) == 0:
                num_latent = {}
                for n, var in enumerate(x_vars):
                    domain = var.get_domain(transform=True)
                    num_latent[var] = len(domain) if isinstance(domain, list) else 0
                self.latent_size = num_latent

                new_pt = {}
                new_grid_size = self.beta_to_knots(beta)
                for n, var in enumerate(x_vars):
                    num_new_pts = new_grid_size[n]
                    domain = var.get_domain(transform=True)
                    if isinstance(domain, list):  # store a grid for each latent coefficient (Nlatent, Ngrid)
                        new_pt[var] = np.atleast_1d([self.collocation_1d(num_new_pts[i], d, method=self.collocation_rule)
                                                     for i, d in enumerate(domain)]).tolist()
                    else:
                        wt_fcn = lambda z: var.pdf(var.denormalize(z), transform=True)
                        new_pt[var] = var.denormalize(self.collocation_1d(num_new_pts, domain, wt_fcn=wt_fcn,
                                                                          method=self.collocation_rule)).tolist()
                self.x_grids = new_pt
            self.betas.add(beta)
            self.yi_map.setdefault(alpha, dict())
            self.yi_nan_map.setdefault(alpha, dict())
            self.error_map.setdefault(alpha, dict())
            return list(self._expand_grid_coords(beta)), self._append_grid_points(beta)

        # Otherwise, refine the sparse grid
        for beta_old in self.betas:
            # Get the first lower neighbor in the sparse grid and refine the 1d grid
            if self.is_one_level_refinement(beta_old, beta):
                dim_refine = int(np.nonzero(np.array(beta, dtype=int) - np.array(beta_old, dtype=int))[0][0])
                var_refine = x_vars[dim_refine]
                domain = var_refine.get_domain(transform=True)
                new_grid_size = self.beta_to_knots(beta)
                old_grid_size = self.beta_to_knots(beta_old)
                grid_refine = np.atleast_1d(self.x_grids[var_refine])
                num_latent = self.latent_size[var_refine]

                # Handle refining latent grids
                if num_latent > 0:
                    old_latent_size = old_grid_size[dim_refine]  # tuples of latent grid sizes
                    new_latent_size = new_grid_size[dim_refine]
                    if max(new_latent_size) > grid_refine.shape[-1]:
                        grid_refine = np.pad(grid_refine, [(0, 0), (0, max(new_latent_size) - grid_refine.shape[-1])],
                                             mode='constant', constant_values=np.nan)
                    for i in range(num_latent):
                        if old_latent_size[i] < new_latent_size[i]:
                            num_new_pts = new_latent_size[i] - old_latent_size[i]
                            grid_refine[i] = self.collocation_1d(num_new_pts, domain[i],
                                                                 grid_refine[i, :old_latent_size[i]],
                                                                 method=self.collocation_rule)
                    self.x_grids[var_refine] = grid_refine.tolist()

                # Handle scalar refinement per usual
                else:
                    if grid_refine.shape[-1] < new_grid_size[dim_refine]:
                        num_new_pts = new_grid_size[dim_refine] - old_grid_size[dim_refine]
                        grid_refine = var_refine.normalize(grid_refine)
                        wt_fcn = lambda z: var_refine.pdf(var_refine.denormalize(z), transform=True)
                        self.x_grids[var_refine] = np.atleast_1d(
                            var_refine.denormalize(self.collocation_1d(num_new_pts, domain, grid_refine, wt_fcn=wt_fcn,
                                                                       method=self.collocation_rule))).tolist()
                break

        new_coords = []
        new_pts = {}
        for coord in self._expand_grid_coords(beta):
            if coord not in self.yi_map[alpha]:
                # If we have not computed this grid coordinate yet
                self._append_grid_points(coord, new_pts)
                new_coords.append(coord)

        self.betas.add(beta)
        return new_coords, new_pts

    def _append_grid_points(self, coord: tuple, pts: dict = None):
        """Extract the `x` grid point located at `coord` from `x_grids` and append to the `pts` dictionary."""
        if pts is None:
            pts = {}
        for i, var in enumerate(self.x_grids):
            grid = np.atleast_1d(self.x_grids[var])
            if len(grid.shape) == 1:
                new_pt = np.atleast_1d(grid[coord[i]])                          # select scalar grid point
            else:
                new_pt = grid[range(grid.shape[0]), coord[i]].reshape((1, -1))  # select latent coefficients
            pts[var] = new_pt if pts.get(var) is None else np.concatenate((pts[var], new_pt), axis=0)

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
            except:
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
                       wt_fcn: callable = None, method='leja') -> np.ndarray:
        """Find the next `N` points in the 1d sequence of `z_pts` using the provided collocation method.

        :param N: number of new points to add to the sequence
        :param z_bds: bounds on the 1d domain
        :param z_pts: current univariate sequence `(Nz,)`, start at middle of `z_bds` if `None`
        :param wt_fcn: weighting function, uses a constant weight if `None`, callable as `wt_fcn(z)`
        :param method: collocation method to use, currently only 'leja' is supported
        :returns: the univariate sequence `z_pts` augmented by `N` new points
        """
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
                    res = direct(obj_fun, [z_bds])  # Use global DIRECT optimization over 1d domain
                    z_star = res.x
                    z_pts = np.concatenate((z_pts, z_star))
            case other:
                raise NotImplementedError(f"Unknown collocation method: {method}")

        return z_pts
