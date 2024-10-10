"""Classes for storing and managing training data for surrogate models.

Includes:

- `TrainingData` — an interface for storing surrogate training data.
- `SparseGrid` — a class for storing training data in a sparse grid format.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import field, dataclass

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
    def get(self, alpha: MultiIndex, beta: MultiIndex):
        """Return the `xi, yi` training values for a given multi-index pair.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :returns: `dicts` of model input `xi` and output `yi` values
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, alpha: tuple, beta: tuple, coords: list, yi_dict: dict[str, ArrayLike]):
        """Store training data for a given multi-index pair.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param coords: locations for storing the `yi` values in the underlying data structure
        :param yi_dict: a `dict` of model output `yi` values, each entry should be the same length as `coords`
        """
        raise NotImplementedError

    @abstractmethod
    def refine(self, alpha: MultiIndex, beta: MultiIndex, x_vars: VariableList):
        """Return new design points for a given multi-index pair."""
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
    collocation_rule: str = 'leja'
    knots_per_level: int = 2

    betas: set[MultiIndex] = field(default_factory=set)
    x_grids: dict[str, ArrayLike] = field(default_factory=dict)
    xi_map: dict[tuple[int, ...], dict[str, ArrayLike]] = field(default_factory=dict)
    yi_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, ArrayLike]]] = field(default_factory=dict)
    yi_nan_map: dict[MultiIndex, dict[tuple[int, ...], dict[str, ArrayLike]]] = field(default_factory=dict)

    def get(self, alpha: MultiIndex, beta: MultiIndex):
        """Get the training data from the sparse grid for a given `alpha` and `beta` pair."""
        xi_dict = {}
        yi_dict = {}
        grid_size = self.beta_to_knots(beta, knots_per_level=self.knots_per_level)
        grid_coords = [list(range(s)) for s in grid_size]
        for coord in itertools.product(*grid_coords):
            try:
                yi_curr = self.yi_map[alpha][coord]
                if any([np.any(np.isnan(yi)) for yi in yi_curr.values()]):
                    yi_curr = self.yi_nan_map[alpha].get(coord, yi_curr)
                if not any([np.any(np.isnan(yi)) for yi in yi_curr.values()]):
                    for var in self.x_grids.keys():
                        xi_dict[var] = self.xi_map[coord][var] if xi_dict.get(var) is None else np.concatenate(
                            (xi_dict[var], self.xi_map[coord][var]), axis=0)
                    for var in yi_curr.keys():
                        yi_dict[var] = yi_curr if yi_dict.get(var) is None else (
                            np.concatenate((yi_dict[var], yi_curr), axis=0))
            except KeyError as e:
                raise ValueError(f"Can't access sparse grid data for alpha={alpha}, beta={beta}, coord={coord}. "
                                 f"Make sure the data has been set first.") from e

        return xi_dict, yi_dict

    def set(self, alpha: tuple, beta: tuple, coords: list, yi_dict: dict[str, ArrayLike]):
        """Store model output `yi` values, accounting for possible `nans` by regression imputation.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param coords: a list of grid coordinates to locate the `yi` values in the sparse grid data structure
        :param yi_dict: a `dict` of model output `yi` values
        """
        errors = yi_dict.pop('errors', {})
        for i, coord in enumerate(coords):
            self.yi_map[alpha][coord] = {var: yi[i, ...] for var, yi in yi_dict.items()}
            if err_info := errors.get(i):
                self.yi_map[alpha][coord]['errors'] = err_info

        imputer, xi_all, yi_all = None, None, None
        for coord, yi_dict in self.yi_map[alpha].items():
            # Don't try to impute non-ndarrays or special model return values
            output_vars = [var for var in yi_dict.keys() if var not in Dataset.__annotations__.keys()
                           and isinstance(yi_dict[var], np.ndarray)]

            if any([np.any(np.isnan(yi_dict[var])) for var in output_vars]):
                if imputer is None:
                    # Grab all 'good' interpolation points and train a simple linear regression fit
                    xi_all, yi_all = self.get(alpha, beta)
                    xi_mat = np.concatenate([xi_all[var][..., np.newaxis] for var in xi_all.keys()], axis=-1)
                    yi_mat = np.concatenate([yi_all[var][..., np.newaxis] for var in output_vars], axis=-1)

                    imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
                    imputer.fit(xi_mat, yi_mat)
                x_interp = np.concatenate([self.xi_map[coord][var][..., np.newaxis] for var in xi_all.keys()], axis=-1)
                y_interp = imputer.predict(x_interp)
                y_impute = {}
                for i, var in enumerate(output_vars):
                    y_curr = y_interp[..., i]
                    nan_idx = np.isnan(yi_dict[var])
                    y_curr[~nan_idx] = yi_dict[var][~nan_idx]  # Only keep imputed values where yi is nan
                    y_impute[var] = y_curr
                self.yi_nan_map[alpha][coord] = y_impute

    def refine(self, alpha: MultiIndex, beta: MultiIndex, x_vars: VariableList):
        if np.sum(beta) == 0:
            # Initialize a sparse grid with a single collocation point
            new_pt = {var: var.denormalize(self.collocation_1d(1, var.get_domain(transform=True),
                                                               method=self.collocation_rule)) for var in x_vars}
            self.betas.add(beta)
            self.x_grids = new_pt
            self.xi_map = {beta: {var: new_pt[var][0, ...] for var in x_vars}}
            self.yi_map[alpha] = dict()
            self.yi_nan_map[alpha] = dict()
            return new_pt

        # Otherwise, refine the sparse grid
        new_grid_size = self.beta_to_knots(beta, knots_per_level=self.knots_per_level)
        for beta_old in self.betas:
            # Get the first lower neighbor in the sparse grid and refine the 1d grid
            if self.is_one_level_refinement(beta_old, beta):
                dim_refine = int(np.nonzero(np.array(beta, dtype=int) - np.array(beta_old, dtype=int))[0][0])
                var_refine = x_vars[dim_refine]
                old_grid_size = self.beta_to_knots(beta_old, knots_per_level=self.knots_per_level)
                grid_refine = var_refine.normalize(self.x_grids[var_refine])

                if len(grid_refine) < new_grid_size[dim_refine]:
                    num_new_pts = new_grid_size[dim_refine] - old_grid_size[dim_refine]
                    self.x_grids[var_refine] = (
                        var_refine.denormalize(self.collocation_1d(num_new_pts, var_refine.get_domain(transform=True),
                                                                   grid_refine, method=self.collocation_rule)))
                break

        new_coords = []
        new_pts = {}
        grid_coords = [list(range(s)) for s in new_grid_size]
        for coord in itertools.product(*grid_coords):
            if coord not in self.yi_map[alpha]:
                # If we have not computed this grid coordinate yet
                new_pt = {}
                for i, var in enumerate(x_vars):
                    new_pt[var] = self.x_grids[var][coord[i], ...]
                    if new_pts.get(var) is None:
                        new_pts[var] = new_pt[var]
                    else:
                        new_pts[var] = np.concatenate((new_pts[var], new_pt[var]), axis=0)
                new_coords.append(coord)
                self.xi_map[coord] = new_pt

        self.betas.add(beta)
        return new_coords, new_pts

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

    @staticmethod
    def beta_to_knots(beta: MultiIndex, knots_per_level: int = 2) -> list[int]:
        """Convert a `beta` multi-index to the number of knots per dimension in the sparse grid.

        :param beta: refinement level indices
        :param knots_per_level: level-to-grid-size multiplier, i.e. number of new points (or knots) for each beta level
        :returns: the number of knots/points per dimension for the sparse grid
        """
        return [knots_per_level * level + 1 for level in beta]

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
        z_pts = np.atleast_1d(z_pts).astype(np.float32)

        match method:
            case 'leja':
                # Construct Leja sequence by maximizing the Leja objective sequentially
                for i in range(N):
                    obj_fun = lambda z: -wt_fcn(np.array(z).astype(np.float32)) * np.prod(np.abs(z - z_pts))
                    res = direct(obj_fun, [z_bds])  # Use global DIRECT optimization over 1d domain
                    z_star = res.x
                    z_pts = np.concatenate((z_pts, z_star))
            case other:
                raise NotImplementedError(f"Unknown collocation method: {method}")

        return z_pts
