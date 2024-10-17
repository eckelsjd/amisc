"""Provides interpolator classes. Interpolators approximate the input &rarr; output mapping of a model given
a set of training data. The training data consists of input-output pairs, and the interpolator can be
refined with new training data.

Includes:

- `Interpolator`: Abstract class providing basic structure of an interpolator
- `Lagrange`: Concrete implementation for tensor-product barycentric Lagrange interpolation
- `InterpolatorState`: Interface for a dataclass that stores the internal state of an interpolator
- `LagrangeState`: The internal state for a barycentric Lagrange polynomial interpolator
"""
from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from numpy.typing import ArrayLike

from amisc.serialize import Serializable, Base64Serializable, StringSerializable
from amisc.typing import MultiIndex, Dataset
from amisc.variable import Variable, VariableList

__all__ = ["InterpolatorState", "LagrangeState", "Interpolator", "Lagrange"]


class InterpolatorState(Serializable, ABC):
    """Interface for a dataclass that stores the internal state of an interpolator (e.g. weights and biases)."""
    pass


@dataclass
class LagrangeState(InterpolatorState, Base64Serializable):
    """The internal state for a barycentric Lagrange polynomial interpolator."""
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    x_grids: dict[str, np.ndarray] = field(default_factory=dict)

    def __eq__(self, other):
        if isinstance(other, LagrangeState):
            try:
                return all([np.allclose(self.weights[var], other.weights[var]) for var in self.weights]) and \
                    all([np.allclose(self.x_grids[var], other.x_grids[var]) for var in self.x_grids])
            except IndexError:
                return False
        else:
            return False


class Interpolator(Serializable, ABC):
    """Interface for an interpolator object that approximates a model."""

    @abstractmethod
    def refine(self, beta: MultiIndex, training_data: tuple[Dataset, Dataset],
               old_state: InterpolatorState, x_vars: VariableList) -> InterpolatorState:
        """Refine the interpolator state with new training data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: dict | Dataset, state: InterpolatorState,
                training_data: tuple[Dataset, Dataset], x_vars: VariableList) -> Dataset:
        """Interpolate the output of the model at points `x` using the given state and training data."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @classmethod
    def from_dict(cls, config: dict) -> Interpolator:
        """Create an `Interpolator` object from a `dict` configuration."""
        method = config.pop('method', 'lagrange').lower()
        match method:
            case 'lagrange':
                return Lagrange(**config)
            case other:
                raise NotImplementedError(f"Unknown interpolator method: {method}")


@dataclass
class Lagrange(Interpolator, StringSerializable):
    """Implementation of a tensor-product barycentric Lagrange polynomial interpolator."""
    interval_capacity: float = 4.0

    @staticmethod
    def _extend_grids(x_grids: dict[str, np.ndarray], x_points: dict[str, np.ndarray]):
        """Extend the 1d `x` grids with any new points from `x_points`, skipping duplicates."""
        extended_grids = copy.deepcopy(x_grids)
        for var, new_pts in x_points.items():
            if var not in x_grids:
                extended_grids[var] = new_pts
            else:
                # Handle latent variables (dim=2)
                if len(new_pts.shape) > 1:
                    num_latent, old_len = x_grids[var].shape
                    new_grids = [np.concatenate((x_grids[var][i], np.setdiff1d(new_pts[i], x_grids[var][i])), axis=0)
                                 for i in range(num_latent)]
                    new_len = max([len(grid) for grid in new_grids])
                    if new_len > old_len:  # Pad the old grids with NaNs
                        extended_grids[var] = np.pad(x_grids[var], [(0, 0), (0, new_len - old_len)], mode='constant',
                                                     constant_values=np.nan)
                    for i, grid in enumerate(new_grids):
                        extended_grids[var][i, :len(grid)] = grid

                # Handle regular scalar variables (dim=1)
                else:
                    extended_grids[var] = np.concatenate((x_grids[var], np.setdiff1d(new_pts, x_grids[var])), axis=0)
        return extended_grids

    def refine(self, beta: MultiIndex, training_data: tuple[Dataset, Dataset],
               old_state: LagrangeState, x_vars: VariableList) -> LagrangeState:
        """Refine the interpolator state with new training data.

        :param beta: the refinement level indices for the interpolator (not used for Lagrange)
        :param training_data: a tuple of dictionaries containing the new training data (`xtrain`, `ytrain`)
        :param old_state: the old interpolator state to refine
        :param x_vars: the list of variables that define the input domain
        :returns: the new interpolator state
        """
        xtrain, ytrain = training_data  # Lagrange only really needs the xtrain data to update barycentric weights/grids

        # Initialize the interpolator state
        if old_state is None:
            x_grids = self._extend_grids({}, xtrain)
            weights = {}
            for var, grid in x_grids.items():
                bds = x_vars[var].get_domain(transform=True)
                if isinstance(bds, list):   # latent coefficients
                    weights[var] = np.full(grid.shape, np.nan)
                    for i, bd in enumerate(bds):
                        C = (bd[1] - bd[0]) / self.interval_capacity
                        xj = grid[i, ~np.isnan(grid[i]), np.newaxis]
                        Nx = xj.shape[0]
                        xi = xj.reshape((1, Nx))
                        dist = (xj - xi) / C
                        np.fill_diagonal(dist, 1)
                        weights[var][i, :Nx] = (1.0 / np.prod(dist, axis=1))
                else:                       # scalars
                    Nx = grid.shape[0]
                    C = (bds[1] - bds[0]) / self.interval_capacity  # Interval capacity (see Berrut and Trefethen 2004)
                    xj = var.normalize(grid.reshape((Nx, 1)))
                    xi = xj.reshape((1, Nx))
                    dist = (xj - xi) / C
                    np.fill_diagonal(dist, 1)  # Ignore product when i==j
                    weights[var] = (1.0 / np.prod(dist, axis=1))  # (Nx,)

        # Otherwise, refine the interpolator state
        else:
            x_grids = self._extend_grids(old_state.x_grids, xtrain)
            weights = copy.deepcopy(old_state.weights)
            for var, grid in x_grids.items():
                bds = x_vars[var].get_domain(transform=True)

                # Update latent coefficient weights
                if isinstance(bds, list):
                    old_len = old_state.x_grids[var].shape[1]
                    new_len = grid.shape[1]
                    weights[var] = np.pad(weights[var], [(0, 0), (0, new_len - old_len)], mode='constant',
                                          constant_values=np.nan)
                    for i, bd in enumerate(bds):
                        old_grid = old_state.x_grids[var][i, ~np.isnan(old_state.x_grids[var][i])]
                        Nx_old = old_grid.shape[0]
                        Nx_new = grid[i, ~np.isnan(grid[i])].shape[0]
                        if Nx_new > Nx_old:
                            C = (bd[1] - bd[0]) / self.interval_capacity
                            xi = grid[i, :Nx_new]
                            for j in range(Nx_old, Nx_new):
                                weights[var][i, :j] *= (C / (xi[:j] - xi[j]))
                                weights[var][i, j] = np.prod(C / (xi[j] - xi[:j]))
                # Update scalar weights
                else:
                    Nx_old = old_state.x_grids[var].shape[0]
                    Nx_new = grid.shape[0]
                    if Nx_new > Nx_old:
                        new_wts = weights[var]
                        C = (bds[1] - bds[0]) / self.interval_capacity
                        xi = var.normalize(grid)
                        for j in range(Nx_old, Nx_new):
                            new_wts[:j] *= (C / (xi[:j] - xi[j]))
                            new_wts[j] = np.prod(C / (xi[j] - xi[:j]))

        return LagrangeState(weights=weights, x_grids=x_grids)

    def predict(self, x: Dataset, state: LagrangeState, training_data, x_vars: VariableList):
        """Predict the output of the model at points `x` with barycentric Lagrange interpolation."""
        # Convert `x` and `yi` to 2d arrays: (N, xdim) and (N, ydim)
        _, yi = training_data
        x_arr = np.empty((next(iter(x.values())).shape[:-1], 0))
        yi_arr = np.empty((next(iter(yi.values())).shape[:-1], 0))
        for var in x_vars:
            x_arr = np.concatenate((x_arr, x[var]), axis=-1) if len(var.shape) > 0 else (
                np.concatenate((x_arr, var.normalize(x[var][..., np.newaxis])), axis=-1))
        for var in yi:
            yi_arr = np.concatenate((yi_arr, yi[var]), axis=-1) if len(yi[var].shape) > 1 else (
                np.concatenate((yi_arr, yi[var][..., np.newaxis]), axis=-1))

        xdim = x_arr.shape[-1]
        ydim = yi_arr.shape[-1]
        grid_sizes = {var: grid.shape[-1] for var, grid in state.x_grids.items()}
        max_size = max(grid_sizes.values())
        dims = list(range(xdim))

        # Create ragged edge matrix of interpolation pts and weights
        x_j = np.full((xdim, max_size), np.nan)                             # For example:
        w_j = np.full((xdim, max_size), np.nan)                             # A= [#####--
        n = 0                                                               #     #######
        for var in x_vars:                                                  #     ###----]
            if len(state.x_grids[var].shape) > 1:  # latent
                for grid, weights in zip(state.x_grids[var], state.weights[var]):
                    x_j[n, :grid_sizes[var]] = grid
                    w_j[n, :grid_sizes[var]] = weights
                    n += 1
            else:                                   # scalars
                x_j[n, :grid_sizes[var]] = var.normalize(state.x_grids[var])
                w_j[n, :grid_sizes[var]] = state.weights[var]
                n += 1

        diff = x_arr[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        diff[div_zero_idx] = 1
        quotient = w_j / diff                           # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)             # (..., xdim)
        y = np.zeros(x_arr.shape[:-1] + (ydim,))        # (..., ydim)

        # Loop over multi-indices and compute tensor-product lagrange polynomials
        indices = [range(np.count_nonzero(~np.isnan(grid))) for grid in x_j]
        for i, j in enumerate(itertools.product(*indices)):
            L_j = quotient[..., dims, j] / qsum         # (..., xdim)
            other_pts = np.copy(div_zero_idx)
            other_pts[div_zero_idx[..., dims, j]] = False

            # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
            L_j[div_zero_idx[..., dims, j]] = 1
            L_j[np.any(other_pts, axis=-1)] = 0

            # Add multivariate basis polynomial contribution to interpolation output
            L_j = np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)
            y += L_j * yi_arr[i, :]

        # Unpack the outputs back into a Dataset
        y_ret = {}
        start_idx = 0
        for var, arr in yi.items():
            num_vals = arr.shape[-1] if len(arr.shape) > 1 else 1
            end_idx = start_idx + num_vals
            y_ret[var] = y[..., start_idx:end_idx]
            if len(arr.shape) == 1:
                y_ret[var] = np.squeeze(y_ret[var], axis=-1)  # for scalars
            start_idx = end_idx
        return y_ret


class BaseInterpolator(ABC):
    """Base interpolator abstract class.

    !!! Info "Setting the training data"
        You can leave the training data `xi`, `yi` empty; they can be iteratively refined later on.

    !!! Info "Model specification"
        The model is a callable function of the form `ret = model(x, *args, **kwargs)`. The return value is a dictionary
        of the form `ret = {'y': y, 'files': files, 'cost': cost}`. In the return dictionary, you specify the raw model
        output `y` as an `np.ndarray` at a _minimum_. Optionally, you can specify paths to output files and the average
        model cost (in units of seconds of cpu time), and anything else you want.

    :ivar beta: specifies the refinement level of this interpolator as a set of natural number indices
    :ivar x_vars: list of variables that fully determines the input domain of interest for interpolation
    :ivar xi: `(Nx, x_dim)`, interpolation points (or knots, training samples, etc.) stored as an array
    :ivar yi: `(Nx, y_dim)`, function values at the interpolation points, i.e. the training data
    :ivar _model: stores a ref to the model or function that is to be interpolated, callable as `ret = model(x)`
    :ivar _model_args: additional arguments to supply to the model
    :ivar _model_kwargs: additional keyword arguments to supply to the model
    :ivar model_cost: the average total cpu time (in seconds) for a single model evaluation call of one set of inputs
    :ivar output_files: tracks model output files corresponding to `yi` training data (for more complex models)
    :ivar logger: a logging utility reference

    :vartype beta: tuple[int, ...]
    :vartype x_vars: list[Variable]
    :vartype xi: np.ndarray
    :vartype yi: np.ndarray
    :vartype _model: callable[np.ndarray] -> dict
    :vartype _model_args: tuple
    :vartype _model_kwargs: dict
    :vartype model_cost: float
    :vartype output_files: list[str | Path]
    :vartype logger: logging.Logger
    """

    def __init__(self, beta: tuple, x_vars: Variable | list[Variable], xi=None, yi=None,
                 model=None, model_args=(), model_kwargs=None):
        """Construct the interpolator.

        :param beta: refinement level indices
        :param x_vars: list of variables to specify input domain of interpolation
        :param xi: `(Nx, xdim)`, interpolation points (optional)
        :param yi: `(Nx, ydim)`, the function values at the interpolation points (optional)
        :param model: callable as {'y': y} = model(x), with `x = (..., x_dim)`, `y = (..., y_dim)`
        :param model_args: optional args for the model
        :param model_kwargs: optional kwargs for the model
        """
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate the interpolation at points `x`.

        :param x: `(..., x_dim)`, the points to be interpolated, must be within the input domain for accuracy
        :returns y: `(..., y_dim)`, the interpolated function values
        """
        pass

    @abstractmethod
    def grad(self, x: np.ndarray | float | list, xi: np.ndarray = None, yi: np.ndarray = None) -> np.ndarray:
        """Evaluate the gradient/Jacobian at points `x` using the interpolator.

        :param x: `(..., xdim)`, the evaluation points, must be within domain of `self.xi` for accuracy
        :param xi: `(Ni, xdim)` optional, interpolation grid points to use (e.g. if `self.reduced=True`)
        :param yi: `(Ni, ydim)` optional, function values at xi to use (e.g. if `self.reduced=True`)
        :returns jac: `(..., ydim, xdim)`, the Jacobian at points `x`
        """

    @abstractmethod
    def hessian(self, x: np.ndarray | float | list, xi: np.ndarray = None, yi: np.ndarray = None) -> np.ndarray:
        """Evaluate the Hessian at points `x` using the interpolator.

        :param x: `(..., xdim)`, the evaluation points, must be within domain of `self.xi` for accuracy
        :param xi: `(Ni, xdim)` optional, interpolation grid points to use (e.g. if `self.reduced=True`)
        :param yi: `(Ni, ydim)` optional, function values at xi to use (e.g. if `self.reduced=True`)
        :returns hess: `(..., ydim, xdim, xdim)`, the Hessian at points `x`
        """


class LagrangeInterpolator(BaseInterpolator):
    """Tensor-product (multivariate) grid interpolator, based on barycentric Lagrange polynomials.

    !!! Info
        The refinement level indices `beta` are used in this class to specify anisotropic refinement along each
        coordinate direction of the input domain, so `x_dim = len(beta)`.

    :ivar x_grids: univariate Leja sequence points in each 1d dimension
    :ivar weights: the barycentric weights corresponding to `x_grids`
    :ivar reduced: whether to store `xi` and `yi` training data, can set to `False` to save memory, e.g. if an external
                   sparse grid data structure manages this data instead

    :vartype x_grids: list[np.ndarray]
    :vartype weights: list[np.ndarray]
    :vartype reduced: bool
    """

    def __init__(self, beta: tuple, x_vars: Variable | list[Variable], init_grids=True, reduced=False, **kwargs):
        """Initialize a Lagrange tensor-product grid interpolator.

        :param beta: refinement level indices for each input dimension
        :param x_vars: list of variables specifying bounds/pdfs for each input x
        :param init_grids: whether to compute 1d Leja sequences on initialization
        :param reduced: whether to store xi/yi matrices, e.g. set true if storing in external sparse grid structure
        :param **kwargs: other optional arguments (see `BaseInterpolator`)
        """
        self.weights = []   # Barycentric weights for each dimension
        self.x_grids = []   # Univariate nested leja sequences in each dimension
        self.reduced = reduced
        super().__init__(beta, x_vars, **kwargs)

        if init_grids:
            # Construct 1d univariate Leja sequences in each dimension
            grid_sizes = self.get_grid_sizes(self.beta)
            self.x_grids = [self.leja_1d(grid_sizes[n], self.x_vars[n].get_domain(),
                                         wt_fcn=self.x_vars[n].pdf).astype(np.float32) for n in range(self.xdim())]

            for n in range(self.xdim()):
                Nx = grid_sizes[n]
                bds = self.x_vars[n].get_domain()
                grid = self.x_grids[n]
                C = (bds[1] - bds[0]) / 4.0  # Interval capacity (see Berrut and Trefethen 2004)
                xj = grid.reshape((Nx, 1))
                xi = grid.reshape((1, Nx))
                dist = (xj - xi) / C
                np.fill_diagonal(dist, 1)  # Ignore product when i==j
                self.weights.append((1.0 / np.prod(dist, axis=1)).astype(np.float32))  # (Nx,)

            # Cartesian product of univariate grids
            if not self.reduced:
                self.xi = np.empty((np.prod(grid_sizes), self.xdim()), dtype=np.float32)
                for i, ele in enumerate(itertools.product(*self.x_grids)):
                    self.xi[i, :] = ele

    def refine(self, beta, auto=True, x_refine: np.ndarray = None):
        """Return a new interpolator with one dimension refined by one level, specified by `beta`.

        !!! Note
            If `self.reduced=True` or `auto=False`, then this function will return tuple indices `idx` corresponding
            to the new interpolation points `x`. The tuple indices specify one index along each input dimension.

        :param beta: the new refinement level indices
        :param auto: whether to automatically compute model at refinement points
        :param x_refine: `(Nx,)` use this array as the refined 1d grid if provided, otherwise compute via `leja_1d`
        :returns: `interp` - a `LagrangeInterpolator` with a refined grid (default), otherwise if `auto=False`,
                  returns `idx, x, interp`, where `idx` and `x` correspond to new interpolation points.
        """
        try:
            # Initialize a new interpolator with the new refinement levels
            interp = LagrangeInterpolator(beta, self.x_vars, model=self._model, model_args=self._model_args,
                                          model_kwargs=self._model_kwargs, init_grids=False, reduced=self.reduced)

            # Find the dimension and number of new points to add
            old_grid_sizes = self.get_grid_sizes(self.beta)
            new_grid_sizes = interp.get_grid_sizes(beta)
            dim_refine = 0
            num_new_pts = 0
            for idx, grid_size in enumerate(new_grid_sizes):
                if grid_size != old_grid_sizes[idx]:
                    dim_refine = idx
                    num_new_pts = grid_size - old_grid_sizes[idx]
                    break

            # Add points to leja grid in this dimension
            interp.x_grids = copy.deepcopy(self.x_grids)
            xi = copy.deepcopy(x_refine) if x_refine is not None else self.leja_1d(num_new_pts,
                                                                                   interp.x_vars[dim_refine].get_domain(),
                                                                                   z_pts=interp.x_grids[dim_refine],
                                                                                   wt_fcn=interp.x_vars[dim_refine].pdf)
            interp.x_grids[dim_refine] = xi.astype(np.float32)

            # Update barycentric weights in this dimension
            interp.weights = copy.deepcopy(self.weights)
            Nx_old = old_grid_sizes[dim_refine]
            Nx_new = new_grid_sizes[dim_refine]
            old_wts = copy.deepcopy(self.weights[dim_refine])
            new_wts = np.zeros(Nx_new, dtype=np.float32)
            new_wts[:Nx_old] = old_wts
            bds = interp.x_vars[dim_refine].get_domain()
            C = (bds[1] - bds[0]) / 4.0  # Interval capacity
            xi = interp.x_grids[dim_refine]
            for j in range(Nx_old, Nx_new):
                new_wts[:j] *= (C / (xi[:j] - xi[j]))
                new_wts[j] = np.prod(C / (xi[j] - xi[:j]))
            interp.weights[dim_refine] = new_wts

            # Copy yi over at existing interpolation points
            x_new = np.zeros((0, interp.xdim()), dtype=np.float32)
            x_new_idx = []
            tol = 1e-12     # Tolerance for floating point comparison
            j = 0           # Use this idx for iterating over existing yi
            if not self.reduced:
                interp.xi = np.zeros((np.prod(new_grid_sizes), self.xdim()), dtype=np.float32)
                interp.yi = np.zeros((np.prod(new_grid_sizes), self.ydim()), dtype=np.float32)
                if self.save_enabled():
                    interp.output_files = [None] * np.prod(new_grid_sizes)

            old_indices = [list(range(old_grid_sizes[n])) for n in range(self.xdim())]
            old_indices = list(itertools.product(*old_indices))
            new_indices = [list(range(new_grid_sizes[n])) for n in range(self.xdim())]
            new_indices = list(itertools.product(*new_indices))
            for i in range(len(new_indices)):
                # Get the new grid coordinate/index and physical x location/point
                new_x_idx = new_indices[i]
                new_x_pt = np.array([float(interp.x_grids[n][new_x_idx[n]]) for n in range(self.xdim())],
                                    dtype=np.float32)

                if not self.reduced:
                    # Store the old xi/yi and return new x points
                    interp.xi[i, :] = new_x_pt
                    if j < len(old_indices) and np.all(np.abs(np.array(old_indices[j]) -
                                                              np.array(new_indices[i])) < tol):
                        # If we already have this interpolation point
                        interp.yi[i, :] = self.yi[j, :]
                        if self.save_enabled():
                            interp.output_files[i] = self.output_files[j]
                        j += 1
                    else:
                        # Otherwise, save new interpolation point and its index
                        x_new = np.concatenate((x_new, new_x_pt.reshape((1, self.xdim()))))
                        x_new_idx.append(i)
                else:
                    # Just find the new x indices and return those for the reduced case
                    if j < len(old_indices) and np.all(np.abs(np.array(old_indices[j]) -
                                                              np.array(new_indices[i])) < tol):
                        j += 1
                    else:
                        x_new = np.concatenate((x_new, new_x_pt.reshape((1, self.xdim()))))
                        x_new_idx.append(new_x_idx)     # Add a tuple() multi-index if not saving xi/yi

            # Evaluate the model at new interpolation points
            interp.model_cost = self.model_cost
            if self._model is None:
                self.logger.warning('No model available to evaluate new interpolation points, returning the points '
                                    'to you instead...')
                return x_new_idx, x_new, interp
            elif not auto or self.reduced:
                return x_new_idx, x_new, interp
            else:
                interp.set_yi(x_new=(x_new_idx, x_new))
                return interp
        except Exception as e:
            import traceback
            tb_str = str(traceback.format_exception(e))
            self.logger.error(tb_str)
            raise Exception(f'Original exception in refine(): {tb_str}')

    def __call__(self, x: np.ndarray | float, xi: np.ndarray = None, yi: np.ndarray = None) -> np.ndarray:
        """Evaluate the barycentric interpolation at points `x`.

        :param x: `(..., xdim)`, the points to be interpolated, must be within domain of `self.xi` for accuracy
        :param xi: `(Ni, xdim)` optional, interpolation grid points to use (e.g. if `self.reduced=True`)
        :param yi: `(Ni, ydim)` optional, function values at xi to use (e.g. if `self.reduced=True`)
        :returns y: `(..., ydim)`, the interpolated function values
        """
        shape_1d, x = self._fmt_input(x)
        if yi is None:
            yi = self.yi.copy()
        if xi is None:
            xi = self.xi.copy()
        xdim = xi.shape[-1]
        ydim = yi.shape[-1]
        dims = list(range(xdim))

        nan_idx = np.any(np.isnan(yi), axis=-1)
        if np.any(nan_idx):
            # Use a simple linear regression fit to impute missing values (may have resulted from bad model outputs)
            imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
            imputer.fit(xi[~nan_idx, :], yi[~nan_idx, :])
            yi[nan_idx, :] = imputer.predict(xi[nan_idx, :])

        # Create ragged edge matrix of interpolation pts and weights
        grid_sizes = self.get_grid_sizes(self.beta)     # For example:
        x_j = np.empty((xdim, max(grid_sizes)))         # A= [#####--
        w_j = np.empty((xdim, max(grid_sizes)))               #######
        x_j[:] = np.nan                                       ###----]
        w_j[:] = np.nan
        for n in range(xdim):
            x_j[n, :grid_sizes[n]] = self.x_grids[n]
            w_j[n, :grid_sizes[n]] = self.weights[n]
        diff = x[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        diff[div_zero_idx] = 1
        quotient = w_j / diff                   # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)     # (..., xdim)

        # Loop over multi-indices and compute tensor-product lagrange polynomials
        y = np.zeros(x.shape[:-1] + (ydim,), dtype=x.dtype)    # (..., ydim)
        indices = [np.arange(grid_sizes[n]) for n in range(xdim)]
        for i, j in enumerate(itertools.product(*indices)):
            L_j = quotient[..., dims, j] / qsum  # (..., xdim)
            other_pts = np.copy(div_zero_idx)
            other_pts[div_zero_idx[..., dims, j]] = False

            # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
            L_j[div_zero_idx[..., dims, j]] = 1
            L_j[np.any(other_pts, axis=-1)] = 0

            # Add multivariate basis polynomial contribution to interpolation output
            L_j = np.prod(L_j, axis=-1, keepdims=True)      # (..., 1)
            y += L_j * yi[i, :]

        return np.atleast_1d(np.squeeze(y)) if shape_1d else y

    def grad(self, x: np.ndarray | float | list, xi: np.ndarray = None, yi: np.ndarray = None) -> np.ndarray:
        """Evaluate the gradient/Jacobian at points `x` using the interpolator.

        :param x: `(..., xdim)`, the evaluation points, must be within domain of `self.xi` for accuracy
        :param xi: `(Ni, xdim)` optional, interpolation grid points to use (e.g. if `self.reduced=True`)
        :param yi: `(Ni, ydim)` optional, function values at xi to use (e.g. if `self.reduced=True`)
        :returns jac: `(..., ydim, xdim)`, the Jacobian at points `x`
        """
        shape_1d, x = self._fmt_input(x)
        if yi is None:
            yi = self.yi.copy()
        if xi is None:
            xi = self.xi.copy()
        xdim = xi.shape[-1]
        ydim = yi.shape[-1]
        nan_idx = np.any(np.isnan(yi), axis=-1)
        if np.any(nan_idx):
            # Use a simple linear regression fit to impute missing values (may have resulted from bad model outputs)
            imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
            imputer.fit(xi[~nan_idx, :], yi[~nan_idx, :])
            yi[nan_idx, :] = imputer.predict(xi[nan_idx, :])

        # Create ragged edge matrix of interpolation pts and weights
        grid_sizes = self.get_grid_sizes(self.beta)     # For example:
        x_j = np.empty((xdim, max(grid_sizes)))         # A= [#####--
        w_j = np.empty((xdim, max(grid_sizes)))               #######
        x_j[:] = np.nan                                       ###----]
        w_j[:] = np.nan
        for n in range(xdim):
            x_j[n, :grid_sizes[n]] = self.x_grids[n]
            w_j[n, :grid_sizes[n]] = self.weights[n]

        # Compute values ahead of time that will be needed for the gradient
        diff = x[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        diff[div_zero_idx] = 1
        quotient = w_j / diff                           # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)             # (..., xdim)
        sqsum = np.nansum(w_j / diff ** 2, axis=-1)     # (..., xdim)

        # Loop over multi-indices and compute derivative of tensor-product lagrange polynomials
        jac = np.zeros(x.shape[:-1] + (ydim, xdim), dtype=x.dtype)  # (..., ydim, xdim)
        indices = [np.arange(grid_sizes[n]) for n in range(self.xdim())]
        for k in range(xdim):
            dims = [idx for idx in np.arange(xdim) if idx != k]
            for i, j in enumerate(itertools.product(*indices)):
                j_dims = [j[idx] for idx in dims]
                L_j = quotient[..., dims, j_dims] / qsum[..., dims]  # (..., xdim-1)
                other_pts = np.copy(div_zero_idx)
                other_pts[div_zero_idx[..., list(np.arange(xdim)), j]] = False

                # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
                L_j[div_zero_idx[..., dims, j_dims]] = 1
                L_j[np.any(other_pts[..., dims, :], axis=-1)] = 0

                # Partial derivative of L_j with respect to x_k
                dLJ_dx = ((w_j[k, j[k]] / (qsum[..., k] * diff[..., k, j[k]])) *
                          (sqsum[..., k] / qsum[..., k] - 1 / diff[..., k, j[k]]))

                # Set derivatives when x is at the interpolation points (i.e. x==x_j)
                p_idx = [idx for idx in np.arange(grid_sizes[k]) if idx != j[k]]
                w_j_large = np.broadcast_to(w_j[k, :], x.shape[:-1] + w_j.shape[-1:]).copy()
                curr_j_idx = div_zero_idx[..., k, j[k]]
                other_j_idx = np.any(other_pts[..., k, :], axis=-1)
                dLJ_dx[curr_j_idx] = -np.nansum((w_j[k, p_idx] / w_j[k, j[k]]) /
                                                (x[curr_j_idx, k, np.newaxis] - x_j[k, p_idx]), axis=-1)
                dLJ_dx[other_j_idx] = ((w_j[k, j[k]] / w_j_large[other_pts[..., k, :]]) /
                                       (x[other_j_idx, k] - x_j[k, j[k]]))

                dLJ_dx = np.expand_dims(dLJ_dx, axis=-1) * np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)

                # Add contribution to the Jacobian
                jac[..., k] += dLJ_dx * yi[i, :]

        return np.atleast_1d(np.squeeze(jac)) if shape_1d else jac

    def hessian(self, x: np.ndarray | float | list, xi: np.ndarray = None, yi: np.ndarray = None) -> np.ndarray:
        """Evaluate the Hessian at points `x` using the interpolator.

        :param x: `(..., xdim)`, the evaluation points, must be within domain of `self.xi` for accuracy
        :param xi: `(Ni, xdim)` optional, interpolation grid points to use (e.g. if `self.reduced=True`)
        :param yi: `(Ni, ydim)` optional, function values at xi to use (e.g. if `self.reduced=True`)
        :returns hess: `(..., ydim, xdim, xdim)`, the vector Hessian at points `x`
        """
        shape_1d, x = self._fmt_input(x)
        if yi is None:
            yi = self.yi.copy()
        if xi is None:
            xi = self.xi.copy()
        xdim = xi.shape[-1]
        ydim = yi.shape[-1]
        nan_idx = np.any(np.isnan(yi), axis=-1)
        if np.any(nan_idx):
            # Use a simple linear regression fit to impute missing values (may have resulted from bad model outputs)
            imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
            imputer.fit(xi[~nan_idx, :], yi[~nan_idx, :])
            yi[nan_idx, :] = imputer.predict(xi[nan_idx, :])

        # Create ragged edge matrix of interpolation pts and weights
        grid_sizes = self.get_grid_sizes(self.beta)     # For example:
        x_j = np.empty((xdim, max(grid_sizes)))         # A= [#####--
        w_j = np.empty((xdim, max(grid_sizes)))               #######
        x_j[:] = np.nan                                       ###----]
        w_j[:] = np.nan
        for n in range(xdim):
            x_j[n, :grid_sizes[n]] = self.x_grids[n]
            w_j[n, :grid_sizes[n]] = self.weights[n]

        # Compute values ahead of time that will be needed for the gradient
        diff = x[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        diff[div_zero_idx] = 1
        quotient = w_j / diff                               # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)                 # (..., xdim)
        qsum_p = -np.nansum(w_j / diff ** 2, axis=-1)       # (..., xdim)
        qsum_pp = 2 * np.nansum(w_j / diff ** 3, axis=-1)   # (..., xdim)

        # Loop over multi-indices and compute 2nd derivative of tensor-product lagrange polynomials
        hess = np.zeros(x.shape[:-1] + (ydim, xdim, xdim), dtype=x.dtype)  # (..., ydim, xdim, xdim)
        indices = [np.arange(grid_sizes[n]) for n in range(self.xdim())]
        for m in range(xdim):
            for n in range(m, xdim):
                dims = [idx for idx in np.arange(xdim) if idx not in [m, n]]
                for i, j in enumerate(itertools.product(*indices)):
                    j_dims = [j[idx] for idx in dims]
                    L_j = quotient[..., dims, j_dims] / qsum[..., dims]  # (..., xdim-2)
                    other_pts = np.copy(div_zero_idx)
                    other_pts[div_zero_idx[..., list(np.arange(xdim)), j]] = False

                    # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
                    L_j[div_zero_idx[..., dims, j_dims]] = 1
                    L_j[np.any(other_pts[..., dims, :], axis=-1)] = 0

                    # Cross-terms in Hessian
                    if m != n:
                        # Partial derivative of L_j with respect to x_m and x_n
                        d2LJ_dx2 = np.ones(x.shape[:-1])
                        for k in [m, n]:
                            dLJ_dx = ((w_j[k, j[k]] / (qsum[..., k] * diff[..., k, j[k]])) *
                                      (-qsum_p[..., k] / qsum[..., k] - 1 / diff[..., k, j[k]]))

                            # Set derivatives when x is at the interpolation points (i.e. x==x_j)
                            p_idx = [idx for idx in np.arange(grid_sizes[k]) if idx != j[k]]
                            w_j_large = np.broadcast_to(w_j[k, :], x.shape[:-1] + w_j.shape[-1:]).copy()
                            curr_j_idx = div_zero_idx[..., k, j[k]]
                            other_j_idx = np.any(other_pts[..., k, :], axis=-1)
                            dLJ_dx[curr_j_idx] = -np.nansum((w_j[k, p_idx] / w_j[k, j[k]]) /
                                                            (x[curr_j_idx, k, np.newaxis] - x_j[k, p_idx]), axis=-1)
                            dLJ_dx[other_j_idx] = ((w_j[k, j[k]] / w_j_large[other_pts[..., k, :]]) /
                                                   (x[other_j_idx, k] - x_j[k, j[k]]))

                            d2LJ_dx2 *= dLJ_dx

                        d2LJ_dx2 = np.expand_dims(d2LJ_dx2, axis=-1) * np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)
                        hess[..., m, n] += d2LJ_dx2 * yi[i, :]
                        hess[..., n, m] += d2LJ_dx2 * yi[i, :]

                    # Diagonal terms in Hessian:
                    else:
                        front_term = w_j[m, j[m]] / (qsum[..., m] * diff[..., m, j[m]])
                        first_term = (-qsum_pp[..., m] / qsum[..., m]) + 2*(qsum_p[..., m] / qsum[..., m]) ** 2
                        second_term = (2*(qsum_p[..., m] / (qsum[..., m] * diff[..., m, j[m]]))
                                       + 2 / diff[..., m, j[m]] ** 2)
                        d2LJ_dx2 = front_term * (first_term + second_term)

                        # Set derivatives when x is at the interpolation points (i.e. x==x_j)
                        curr_j_idx = div_zero_idx[..., m, j[m]]
                        other_j_idx = np.any(other_pts[..., m, :], axis=-1)
                        if np.any(curr_j_idx) or np.any(other_j_idx):
                            p_idx = [idx for idx in np.arange(grid_sizes[m]) if idx != j[m]]
                            w_j_large = np.broadcast_to(w_j[m, :], x.shape[:-1] + w_j.shape[-1:]).copy()
                            x_j_large = np.broadcast_to(x_j[m, :], x.shape[:-1] + x_j.shape[-1:]).copy()

                            # if these points are at the current j interpolation point
                            d2LJ_dx2[curr_j_idx] = (2 * np.nansum((w_j[m, p_idx] / w_j[m, j[m]]) /
                                                                 (x[curr_j_idx, m, np.newaxis] - x_j[m, p_idx]), axis=-1) ** 2 +  # noqa: E501
                                                    2 * np.nansum((w_j[m, p_idx] / w_j[m, j[m]]) /
                                                                  (x[curr_j_idx, m, np.newaxis] - x_j[m, p_idx])**2, axis=-1))  # noqa: E501

                            # if these points are at any other interpolation point
                            other_pts_inv = other_pts.copy()
                            other_pts_inv[other_j_idx, m, :grid_sizes[m]] = np.invert(other_pts[other_j_idx, m, :grid_sizes[m]])  # noqa: E501
                            curr_x_j = x_j_large[other_pts[..., m, :]].reshape((-1, 1))
                            other_x_j = x_j_large[other_pts_inv[..., m, :]].reshape((-1, len(p_idx)))
                            curr_w_j = w_j_large[other_pts[..., m, :]].reshape((-1, 1))
                            other_w_j = w_j_large[other_pts_inv[..., m, :]].reshape((-1, len(p_idx)))
                            curr_div = w_j[m, j[m]] / np.squeeze(curr_w_j, axis=-1)
                            curr_diff = np.squeeze(curr_x_j, axis=-1) - x_j[m, j[m]]
                            d2LJ_dx2[other_j_idx] = ((-2*curr_div / curr_diff) * (np.nansum(
                                (other_w_j / curr_w_j) / (curr_x_j - other_x_j), axis=-1) + 1 / curr_diff))

                        d2LJ_dx2 = np.expand_dims(d2LJ_dx2, axis=-1) * np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)
                        hess[..., m, n] += d2LJ_dx2 * yi[i, :]

        return np.atleast_1d(np.squeeze(hess)) if shape_1d else hess
