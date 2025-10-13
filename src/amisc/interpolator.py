"""Provides interpolator classes. Interpolators approximate the input &rarr; output mapping of a model given
a set of training data. The training data consists of input-output pairs, and the interpolator can be
refined with new training data.

Includes:

- `Interpolator`: Abstract class providing basic structure of an interpolator
- `Lagrange`: Concrete implementation for tensor-product barycentric Lagrange interpolation
- `Linear`: Concrete implementation for linear regression using `sklearn`
- `GPR`: Concrete implementation for Gaussian process regression using `sklearn`
- `InterpolatorState`: Interface for a dataclass that stores the internal state of an interpolator
- `LagrangeState`: The internal state for a barycentric Lagrange polynomial interpolator
- `LinearState`: The internal state for a linear interpolator (using sklearn)
- `GPRState`: The internal state for a Gaussian process regression interpolator (using sklearn)
"""
from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from amisc.serialize import Base64Serializable, Serializable, StringSerializable
from amisc.typing import Dataset, MultiIndex

__all__ = ["InterpolatorState", "LagrangeState", "LinearState", "GPRState", "Interpolator", "Lagrange", "Linear", "GPR"]


class InterpolatorState(Serializable, ABC):
    """Interface for a dataclass that stores the internal state of an interpolator (e.g. weights and biases)."""
    pass


@dataclass
class LagrangeState(InterpolatorState, Base64Serializable):
    """The internal state for a barycentric Lagrange polynomial interpolator.

    :ivar weights: the 1d interpolation grid weights
    :ivar x_grids: the 1d interpolation grids
    """
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


@dataclass
class LinearState(InterpolatorState, Base64Serializable):
    """The internal state for a linear interpolator (using sklearn).

    :ivar x_vars: the input variables in order
    :ivar y_vars: the output variables in order
    :ivar regressor: the sklearn regressor object, a pipeline that consists of a `PolynomialFeatures` and a model from
                     `sklearn.linear_model`, i.e. Ridge, Lasso, etc.
    """
    x_vars: list[str] = field(default_factory=list)
    y_vars: list[str] = field(default_factory=list)
    regressor: Pipeline = None

    def __eq__(self, other):
        if isinstance(other, LinearState):
            return (self.x_vars == other.x_vars and self.y_vars == other.y_vars and
                    np.allclose(self.regressor['poly'].powers_, other.regressor['poly'].powers_) and
                    np.allclose(self.regressor['linear'].coef_, other.regressor['linear'].coef_) and
                    np.allclose(self.regressor['linear'].intercept_, other.regressor['linear'].intercept_))
        else:
            return False


@dataclass
class GPRState(InterpolatorState, Base64Serializable):
    """The internal state for a Gaussian Process Regressor interpolator (using sklearn).

    :ivar x_vars: the input variables in order
    :ivar y_vars: the output variables in order
    :ivar regressor: the sklearn regressor object, a pipeline that consists of a preprocessing scaler and a
                     `GaussianProcessRegressor`.
    """
    x_vars: list[str] = field(default_factory=list)
    y_vars: list[str] = field(default_factory=list)
    regressor: Pipeline = None

    def __eq__(self, other):
        if isinstance(other, GPRState):
            return (self.x_vars == other.x_vars and self.y_vars == other.y_vars and
                    len(self.regressor.steps) == len(other.regressor.steps) and
                    self.regressor['gpr'].alpha == other.regressor['gpr'].alpha and
                    self.regressor['gpr'].kernel_ == other.regressor['gpr'].kernel_)
        else:
            return False


class Interpolator(Serializable, ABC):
    """Interface for an interpolator object that approximates a model. An interpolator should:

    - `refine` - take an old state and new training data and produce a new "refined" state (e.g. new weights/biases)
    - `predict` - interpolate from the training data to a new set of points (i.e. approximate the underlying model)
    - `gradient` - compute the grdient/Jacobian at new points (if you want)
    - `hessian` - compute the 2nd derivative/Hessian at new points (if you want)

    Currently, `Lagrange`, `Linear`, and `GPR` interpolators are supported and can be constructed from a configuration
    `dict` via `Interpolator.from_dict()`.
    """

    @abstractmethod
    def refine(self, beta: MultiIndex, training_data: tuple[Dataset, Dataset],
               old_state: InterpolatorState, input_domains: dict[str, tuple]) -> InterpolatorState:
        """Refine the interpolator state with new training data.

        :param beta: a multi-index specifying the fidelity "levels" of the new interpolator state (starts at (0,... 0))
        :param training_data: a tuple of `xi, yi` Datasets for the input/output training data
        :param old_state: the previous state of the interpolator (None if initializing the first state)
        :param input_domains: a `dict` mapping input variables to their corresponding domains
        :returns: the new "refined" interpolator state
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: dict | Dataset, state: InterpolatorState, training_data: tuple[Dataset, Dataset]) -> Dataset:
        """Interpolate the output of the model at points `x` using the given state and training data

        :param x: the input Dataset `dict` mapping input variables to locations at which to compute the interpolator
        :param state: the current state of the interpolator
        :param training_data: a tuple of `xi, yi` Datasets for the input/output training data for the current state
        :returns: a Dataset `dict` mapping output variables to interpolator outputs
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def gradient(self, x: Dataset, state: InterpolatorState, training_data: tuple[Dataset, Dataset]) -> Dataset:
        """Evaluate the gradient/Jacobian at points `x` using the interpolator.

        :param x: the input Dataset `dict` mapping input variables to locations at which to evaluate the Jacobian
        :param state: the current state of the interpolator
        :param training_data: a tuple of `xi, yi` Datasets for the input/output training data for the current state
        :returns: a Dataset `dict` mapping output variables to Jacobian matrices of shape `(ydim, xdim)` -- for
                  scalar outputs, the Jacobian is returned as `(xdim,)`
        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, x: Dataset, state: InterpolatorState, training_data: tuple[Dataset, Dataset]) -> Dataset:
        """Evaluate the Hessian at points `x` using the interpolator.

        :param x: the input Dataset `dict` mapping input variables to locations at which to evaluate the Hessian
        :param state: the current state of the interpolator
        :param training_data: a tuple of `xi, yi` Datasets for the input/output training data for the current state
        :returns: a Dataset `dict` mapping output variables to Hessian matrices of shape `(xdim, xdim)`
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, config: dict) -> Interpolator:
        """Create an `Interpolator` object from a `dict` config. Available methods are `lagrange`, `linear`, and `gpr`.
        Will attempt to find the method if not listed.

        :param config: a `dict` containing the configuration for the interpolator, with the `method` key specifying the
                       name of the interpolator method to use, and the rest of the keys are options for the method
        """
        method = config.pop('method', 'lagrange')
        match method.lower():
            case 'lagrange':
                return Lagrange(**config)
            case 'linear':
                return Linear(**config)
            case 'gpr':
                return GPR(**config)
            case _:
                import amisc.interpolator

                if hasattr(amisc.interpolator, method):
                    return getattr(amisc.interpolator, method)(**config)

                raise NotImplementedError(f"Unknown interpolator method: {method}")


@dataclass
class Lagrange(Interpolator, StringSerializable):
    """Implementation of a tensor-product barycentric Lagrange polynomial interpolator. A `LagrangeState` stores
    the 1d interpolation grids and weights for each input dimension. `Lagrange` computes the tensor-product
    of 1d Lagrange polynomials to approximate a multi-variate function.

    :ivar interval_capacity: tuning knob for Lagrange interpolation (see Berrut and Trefethen 2004)
    """
    interval_capacity: float = 4.0

    @staticmethod
    def _extend_grids(x_grids: dict[str, np.ndarray], x_points: dict[str, np.ndarray]):
        """Extend the 1d `x` grids with any new points from `x_points`, skipping duplicates. This will preserve the
        order of new points in the extended grid without duplication. This will maintain the same order as
        `SparseGrid.x_grids` if that is the underlying training data structure.

        !!! Example
            ```python
            x = {'x': np.array([0, 1, 2])}
            new_x = {'x': np.array([3, 0, 2, 3, 1, 4])}
            extended_x = Lagrange._extend_grids(x, new_x)
            # gives {'x': np.array([0, 1, 2, 3, 4])}
            ```

        !!! Warning
            This will only work for 1d grids; all `x_grids` should be scalar quantities. Field quantities should
            already be passed in as several separate 1d latent coefficients.

        :param x_grids: the current 1d interpolation grids
        :param x_points: the new points to extend the interpolation grids with
        :returns: the extended grids
        """
        extended_grids = copy.deepcopy(x_grids)
        for var, new_pts in x_points.items():
            # Get unique new values that are not already in the grid (maintain their order; keep only first one)
            u = x_grids[var] if var in x_grids else np.array([])
            u, ind = np.unique(new_pts[~np.isin(new_pts, u)], return_index=True)
            u = u[np.argsort(ind)]
            extended_grids[var] = u if var not in x_grids else np.concatenate((x_grids[var], u), axis=0)

        return extended_grids

    def refine(self, beta: MultiIndex, training_data: tuple[Dataset, Dataset],
               old_state: LagrangeState, input_domains: dict[str, tuple]) -> LagrangeState:
        """Refine the interpolator state with new training data.

        :param beta: the refinement level indices for the interpolator (not used for `Lagrange`)
        :param training_data: a tuple of dictionaries containing the new training data (`xtrain`, `ytrain`)
        :param old_state: the old interpolator state to refine (None if initializing)
        :param input_domains: a `dict` of each input variable's domain; input keys should match `xtrain` keys
        :returns: the new interpolator state
        """
        xtrain, ytrain = training_data  # Lagrange only really needs the xtrain data to update barycentric weights/grids

        # Initialize the interpolator state
        if old_state is None:
            x_grids = self._extend_grids({}, xtrain)
            weights = {}
            for var, grid in x_grids.items():
                bds = input_domains[var]
                Nx = grid.shape[0]
                C = (bds[1] - bds[0]) / self.interval_capacity  # Interval capacity (see Berrut and Trefethen 2004)
                xj = grid.reshape((Nx, 1))
                xi = xj.reshape((1, Nx))
                dist = (xj - xi) / C
                np.fill_diagonal(dist, 1)  # Ignore product when i==j
                weights[var] = (1.0 / np.prod(dist, axis=1))  # (Nx,)

        # Otherwise, refine the interpolator state
        else:
            x_grids = self._extend_grids(old_state.x_grids, xtrain)
            weights = copy.deepcopy(old_state.weights)
            for var, grid in x_grids.items():
                bds = input_domains[var]
                Nx_old = old_state.x_grids[var].shape[0]
                Nx_new = grid.shape[0]
                if Nx_new > Nx_old:
                    weights[var] = np.pad(weights[var], [(0, Nx_new - Nx_old)], mode='constant', constant_values=np.nan)
                    C = (bds[1] - bds[0]) / self.interval_capacity
                    for j in range(Nx_old, Nx_new):
                        weights[var][:j] *= (C / (grid[:j] - grid[j]))
                        weights[var][j] = np.prod(C / (grid[j] - grid[:j]))

        return LagrangeState(weights=weights, x_grids=x_grids)

    def predict(self, x: Dataset, state: LagrangeState, training_data: tuple[Dataset, Dataset]):
        """Predict the output of the model at points `x` with barycentric Lagrange interpolation."""
        # Convert `x` and `yi` to 2d arrays: (N, xdim) and (N, ydim)
        # Inputs `x` may come in unordered, but they should get realigned with the internal `x_grids` state
        xi, yi = training_data
        x_arr = np.concatenate([x[var][..., np.newaxis] for var in xi], axis=-1)
        yi_arr = np.concatenate([yi[var][..., np.newaxis] for var in yi], axis=-1)

        xdim = x_arr.shape[-1]
        ydim = yi_arr.shape[-1]
        grid_sizes = {var: grid.shape[-1] for var, grid in state.x_grids.items()}
        max_size = max(grid_sizes.values())
        dims = list(range(xdim))

        # Create ragged edge matrix of interpolation pts and weights
        x_j = np.full((xdim, max_size), np.nan)                             # For example:
        w_j = np.full((xdim, max_size), np.nan)                             # A= [#####--
        for n, var in enumerate(state.x_grids):                             #     #######
            x_j[n, :grid_sizes[var]] = state.x_grids[var]                   #     ###----]
            w_j[n, :grid_sizes[var]] = state.weights[var]

        diff = x_arr[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        check_interp_pts = np.sum(div_zero_idx) > 0     # whether we are evaluating directly on some interp pts
        diff[div_zero_idx] = 1
        quotient = w_j / diff                           # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)             # (..., xdim)
        y = np.zeros(x_arr.shape[:-1] + (ydim,))        # (..., ydim)

        # Loop over multi-indices and compute tensor-product lagrange polynomials
        indices = [range(s) for s in grid_sizes.values()]
        for i, j in enumerate(itertools.product(*indices)):
            L_j = quotient[..., dims, j] / qsum         # (..., xdim)

            # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
            if check_interp_pts:
                other_pts = np.copy(div_zero_idx)
                other_pts[div_zero_idx[..., dims, j]] = False
                L_j[div_zero_idx[..., dims, j]] = 1
                L_j[np.any(other_pts, axis=-1)] = 0

            # Add multivariate basis polynomial contribution to interpolation output
            y += np.prod(L_j, axis=-1, keepdims=True) * yi_arr[i, :]

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

    def gradient(self, x: Dataset, state: LagrangeState, training_data: tuple[Dataset, Dataset]):
        """Evaluate the gradient/Jacobian at points `x` using the interpolator."""
        # Convert `x` and `yi` to 2d arrays: (N, xdim) and (N, ydim)
        xi, yi = training_data
        x_arr = np.concatenate([x[var][..., np.newaxis] for var in xi], axis=-1)
        yi_arr = np.concatenate([yi[var][..., np.newaxis] for var in yi], axis=-1)

        xdim = x_arr.shape[-1]
        ydim = yi_arr.shape[-1]
        grid_sizes = {var: grid.shape[-1] for var, grid in state.x_grids.items()}
        max_size = max(grid_sizes.values())

        # Create ragged edge matrix of interpolation pts and weights
        x_j = np.full((xdim, max_size), np.nan)                             # For example:
        w_j = np.full((xdim, max_size), np.nan)                             # A= [#####--
        for n, var in enumerate(state.x_grids):                             #     #######
            x_j[n, :grid_sizes[var]] = state.x_grids[var]                   #     ###----]
            w_j[n, :grid_sizes[var]] = state.weights[var]

        # Compute values ahead of time that will be needed for the gradient
        diff = x_arr[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        check_interp_pts = np.sum(div_zero_idx) > 0
        diff[div_zero_idx] = 1
        quotient = w_j / diff                               # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)                 # (..., xdim)
        sqsum = np.nansum(w_j / diff ** 2, axis=-1)         # (..., xdim)
        jac = np.zeros(x_arr.shape[:-1] + (ydim, xdim))     # (..., ydim, xdim)

        # Loop over multi-indices and compute derivative of tensor-product lagrange polynomials
        indices = [range(s) for s in grid_sizes.values()]
        for k, var in enumerate(grid_sizes):
            dims = [idx for idx in range(xdim) if idx != k]
            for i, j in enumerate(itertools.product(*indices)):
                j_dims = [j[idx] for idx in dims]
                L_j = quotient[..., dims, j_dims] / qsum[..., dims]  # (..., xdim-1)

                # Partial derivative of L_j with respect to x_k
                dLJ_dx = ((w_j[k, j[k]] / (qsum[..., k] * diff[..., k, j[k]])) *
                          (sqsum[..., k] / qsum[..., k] - 1 / diff[..., k, j[k]]))

                # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
                if check_interp_pts:
                    other_pts = np.copy(div_zero_idx)
                    other_pts[div_zero_idx[..., list(range(xdim)), j]] = False
                    L_j[div_zero_idx[..., dims, j_dims]] = 1
                    L_j[np.any(other_pts[..., dims, :], axis=-1)] = 0

                    # Set derivatives when x is at the interpolation points (i.e. x==x_j)
                    p_idx = [idx for idx in range(grid_sizes[var]) if idx != j[k]]
                    w_j_large = np.broadcast_to(w_j[k, :], x_arr.shape[:-1] + w_j.shape[-1:]).copy()
                    curr_j_idx = div_zero_idx[..., k, j[k]]
                    other_j_idx = np.any(other_pts[..., k, :], axis=-1)
                    dLJ_dx[curr_j_idx] = -np.nansum((w_j[k, p_idx] / w_j[k, j[k]]) /
                                                    (x_arr[curr_j_idx, k, np.newaxis] - x_j[k, p_idx]), axis=-1)
                    dLJ_dx[other_j_idx] = ((w_j[k, j[k]] / w_j_large[other_pts[..., k, :]]) /
                                           (x_arr[other_j_idx, k] - x_j[k, j[k]]))

                dLJ_dx = np.expand_dims(dLJ_dx, axis=-1) * np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)

                # Add contribution to the Jacobian
                jac[..., k] += dLJ_dx * yi_arr[i, :]

        # Unpack the outputs back into a Dataset (array of length xdim for each y_var giving partial derivatives)
        jac_ret = {}
        start_idx = 0
        for var, arr in yi.items():
            num_vals = arr.shape[-1] if len(arr.shape) > 1 else 1
            end_idx = start_idx + num_vals
            jac_ret[var] = jac[..., start_idx:end_idx, :]  # (..., ydim, xdim)
            if len(arr.shape) == 1:
                jac_ret[var] = np.squeeze(jac_ret[var], axis=-2)  # for scalars: (..., xdim) partial derivatives
            start_idx = end_idx
        return jac_ret

    def hessian(self, x: Dataset, state: LagrangeState, training_data: tuple[Dataset, Dataset]):
        """Evaluate the Hessian at points `x` using the interpolator."""
        # Convert `x` and `yi` to 2d arrays: (N, xdim) and (N, ydim)
        xi, yi = training_data
        x_arr = np.concatenate([x[var][..., np.newaxis] for var in xi], axis=-1)
        yi_arr = np.concatenate([yi[var][..., np.newaxis] for var in yi], axis=-1)

        xdim = x_arr.shape[-1]
        ydim = yi_arr.shape[-1]
        grid_sizes = {var: grid.shape[-1] for var, grid in state.x_grids.items()}
        grid_size_list = list(grid_sizes.values())
        max_size = max(grid_size_list)

        # Create ragged edge matrix of interpolation pts and weights
        x_j = np.full((xdim, max_size), np.nan)                             # For example:
        w_j = np.full((xdim, max_size), np.nan)                             # A= [#####--
        for n, var in enumerate(state.x_grids):                             #     #######
            x_j[n, :grid_sizes[var]] = state.x_grids[var]                   #     ###----]
            w_j[n, :grid_sizes[var]] = state.weights[var]

        # Compute values ahead of time that will be needed for the gradient
        diff = x_arr[..., np.newaxis] - x_j
        div_zero_idx = np.isclose(diff, 0, rtol=1e-4, atol=1e-8)
        check_interp_pts = np.sum(div_zero_idx) > 0
        diff[div_zero_idx] = 1
        quotient = w_j / diff                                       # (..., xdim, Nx)
        qsum = np.nansum(quotient, axis=-1)                         # (..., xdim)
        qsum_p = -np.nansum(w_j / diff ** 2, axis=-1)               # (..., xdim)
        qsum_pp = 2 * np.nansum(w_j / diff ** 3, axis=-1)           # (..., xdim)

        # Loop over multi-indices and compute 2nd derivative of tensor-product lagrange polynomials
        hess = np.zeros(x_arr.shape[:-1] + (ydim, xdim, xdim))      # (..., ydim, xdim, xdim)
        indices = [range(s) for s in grid_size_list]
        for m in range(xdim):
            for n in range(m, xdim):
                dims = [idx for idx in range(xdim) if idx not in [m, n]]
                for i, j in enumerate(itertools.product(*indices)):
                    j_dims = [j[idx] for idx in dims]
                    L_j = quotient[..., dims, j_dims] / qsum[..., dims]  # (..., xdim-2)

                    # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
                    if check_interp_pts:
                        other_pts = np.copy(div_zero_idx)
                        other_pts[div_zero_idx[..., list(range(xdim)), j]] = False
                        L_j[div_zero_idx[..., dims, j_dims]] = 1
                        L_j[np.any(other_pts[..., dims, :], axis=-1)] = 0

                    # Cross-terms in Hessian
                    if m != n:
                        # Partial derivative of L_j with respect to x_m and x_n
                        d2LJ_dx2 = np.ones(x_arr.shape[:-1])
                        for k in [m, n]:
                            dLJ_dx = ((w_j[k, j[k]] / (qsum[..., k] * diff[..., k, j[k]])) *
                                      (-qsum_p[..., k] / qsum[..., k] - 1 / diff[..., k, j[k]]))

                            # Set derivatives when x is at the interpolation points (i.e. x==x_j)
                            if check_interp_pts:
                                p_idx = [idx for idx in range(grid_size_list[k]) if idx != j[k]]
                                w_j_large = np.broadcast_to(w_j[k, :], x_arr.shape[:-1] + w_j.shape[-1:]).copy()
                                curr_j_idx = div_zero_idx[..., k, j[k]]
                                other_j_idx = np.any(other_pts[..., k, :], axis=-1)
                                dLJ_dx[curr_j_idx] = -np.nansum((w_j[k, p_idx] / w_j[k, j[k]]) /
                                                                (x_arr[curr_j_idx, k, np.newaxis] - x_j[k, p_idx]),
                                                                axis=-1)
                                dLJ_dx[other_j_idx] = ((w_j[k, j[k]] / w_j_large[other_pts[..., k, :]]) /
                                                       (x_arr[other_j_idx, k] - x_j[k, j[k]]))

                            d2LJ_dx2 *= dLJ_dx

                        d2LJ_dx2 = np.expand_dims(d2LJ_dx2, axis=-1) * np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)
                        hess[..., m, n] += d2LJ_dx2 * yi_arr[i, :]
                        hess[..., n, m] += d2LJ_dx2 * yi_arr[i, :]

                    # Diagonal terms in Hessian:
                    else:
                        front_term = w_j[m, j[m]] / (qsum[..., m] * diff[..., m, j[m]])
                        first_term = (-qsum_pp[..., m] / qsum[..., m]) + 2 * (qsum_p[..., m] / qsum[..., m]) ** 2
                        second_term = (2 * (qsum_p[..., m] / (qsum[..., m] * diff[..., m, j[m]]))
                                       + 2 / diff[..., m, j[m]] ** 2)
                        d2LJ_dx2 = front_term * (first_term + second_term)

                        # Set derivatives when x is at the interpolation points (i.e. x==x_j)
                        if check_interp_pts:
                            curr_j_idx = div_zero_idx[..., m, j[m]]
                            other_j_idx = np.any(other_pts[..., m, :], axis=-1)
                            if np.any(curr_j_idx) or np.any(other_j_idx):
                                p_idx = [idx for idx in range(grid_size_list[m]) if idx != j[m]]
                                w_j_large = np.broadcast_to(w_j[m, :], x_arr.shape[:-1] + w_j.shape[-1:]).copy()
                                x_j_large = np.broadcast_to(x_j[m, :], x_arr.shape[:-1] + x_j.shape[-1:]).copy()

                                # if these points are at the current j interpolation point
                                d2LJ_dx2[curr_j_idx] = (2 * np.nansum((w_j[m, p_idx] / w_j[m, j[m]]) /
                                                                      (x_arr[curr_j_idx, m, np.newaxis] - x_j[m, p_idx]), # noqa: E501
                                                                      axis=-1) ** 2 +
                                                        2 * np.nansum((w_j[m, p_idx] / w_j[m, j[m]]) /
                                                                      (x_arr[curr_j_idx, m, np.newaxis] - x_j[m, p_idx]) ** 2, # noqa: E501
                                                                      axis=-1))

                                # if these points are at any other interpolation point
                                other_pts_inv = other_pts.copy()
                                other_pts_inv[other_j_idx, m, :grid_size_list[m]] = np.invert(
                                    other_pts[other_j_idx, m, :grid_size_list[m]])  # noqa: E501
                                curr_x_j = x_j_large[other_pts[..., m, :]].reshape((-1, 1))
                                other_x_j = x_j_large[other_pts_inv[..., m, :]].reshape((-1, len(p_idx)))
                                curr_w_j = w_j_large[other_pts[..., m, :]].reshape((-1, 1))
                                other_w_j = w_j_large[other_pts_inv[..., m, :]].reshape((-1, len(p_idx)))
                                curr_div = w_j[m, j[m]] / np.squeeze(curr_w_j, axis=-1)
                                curr_diff = np.squeeze(curr_x_j, axis=-1) - x_j[m, j[m]]
                                d2LJ_dx2[other_j_idx] = ((-2 * curr_div / curr_diff) * (np.nansum(
                                    (other_w_j / curr_w_j) / (curr_x_j - other_x_j), axis=-1) + 1 / curr_diff))

                        d2LJ_dx2 = np.expand_dims(d2LJ_dx2, axis=-1) * np.prod(L_j, axis=-1, keepdims=True)  # (..., 1)
                        hess[..., m, n] += d2LJ_dx2 * yi_arr[i, :]

        # Unpack the outputs back into a Dataset (matrix (xdim, xdim) for each y_var giving 2nd partial derivatives)
        hess_ret = {}
        start_idx = 0
        for var, arr in yi.items():
            num_vals = arr.shape[-1] if len(arr.shape) > 1 else 1
            end_idx = start_idx + num_vals
            hess_ret[var] = hess[..., start_idx:end_idx, :, :]  # (..., ydim, xdim, xdim)
            if len(arr.shape) == 1:
                hess_ret[var] = np.squeeze(hess_ret[var], axis=-3)  # for scalars: (..., xdim, xdim) partial derivatives
            start_idx = end_idx
        return hess_ret


@dataclass
class Linear(Interpolator, StringSerializable):
    """Implementation of linear regression using `sklearn`. The `Linear` interpolator uses a pipeline of
    `PolynomialFeatures` and a linear model from `sklearn.linear_model` to approximate the input-output mapping
    with a linear combination of polynomial features. Defaults to Ridge regression (L2 regularization) with
    polynomials of degree 1 (i.e. normal linear regression).

    :ivar regressor: the scikit-learn linear model to use (e.g. 'Ridge', 'Lasso', 'ElasticNet', etc.).
    :ivar scaler: the scikit-learn preprocessing scaler to use (e.g. 'MinMaxScaler', 'StandardScaler', etc.). If None,
                  no scaling is applied (default).
    :ivar regressor_opts: options to pass to the regressor constructor
                          (see [scikit-learn](https://scikit-learn.org/stable/) documentation).
    :ivar scaler_opts: options to pass to the scaler constructor
    :ivar polynomial_opts: options to pass to the `PolynomialFeatures` constructor (e.g. 'degree', 'include_bias').
    """
    regressor: str = 'Ridge'
    scaler: str = None
    regressor_opts: dict = field(default_factory=dict)
    scaler_opts: dict = field(default_factory=dict)
    polynomial_opts: dict = field(default_factory=lambda: {'degree': 1, 'include_bias': False})

    def __post_init__(self):
        try:
            getattr(linear_model, self.regressor)
        except AttributeError:
            raise ImportError(f"Regressor '{self.regressor}' not found in sklearn.linear_model")

        if self.scaler is not None:
            try:
                getattr(preprocessing, self.scaler)
            except AttributeError:
                raise ImportError(f"Scaler '{self.scaler}' not found in sklearn.preprocessing")

    def refine(self, beta: MultiIndex, training_data: tuple[Dataset, Dataset],
               old_state: LinearState, input_domains: dict[str, tuple]) -> InterpolatorState:
        """Train a new linear regression model.

        :param beta: if not empty, then the first element is the number of degrees to add to the polynomial features.
                     For example, if `beta=(1,)`, then the polynomial degree will be increased by 1. If the degree
                     is already set to 1 in `polynomial_opts` (default), then the new degree will be 2.
        :param training_data: a tuple of dictionaries containing the new training data (`xtrain`, `ytrain`)
        :param old_state: the old linear state to refine (only used to get the order of input/output variables)
        :param input_domains: (not used for `Linear`)
        :returns: the new linear state
        """
        polynomial_opts = self.polynomial_opts.copy()
        degree = polynomial_opts.pop('degree', 1)
        if beta != ():
            degree += beta[0]

        pipe = []
        if self.scaler is not None:
            pipe.append(('scaler', getattr(preprocessing, self.scaler)(**self.scaler_opts)))
        pipe.extend([('poly', PolynomialFeatures(degree=degree, **polynomial_opts)),
                     ('linear', getattr(linear_model, self.regressor)(**self.regressor_opts))])
        regressor = Pipeline(pipe)

        xtrain, ytrain = training_data

        # Get order of variables for inputs and outputs
        if old_state is not None:
            x_vars = old_state.x_vars
            y_vars = old_state.y_vars
        else:
            x_vars = list(xtrain.keys())
            y_vars = list(ytrain.keys())

        # Convert to (N, xdim) and (N, ydim) arrays
        x_arr = np.concatenate([xtrain[var][..., np.newaxis] for var in x_vars], axis=-1)
        y_arr = np.concatenate([ytrain[var][..., np.newaxis] for var in y_vars], axis=-1)

        regressor.fit(x_arr, y_arr)

        return LinearState(regressor=regressor, x_vars=x_vars, y_vars=y_vars)

    def predict(self, x: Dataset, state: LinearState, training_data: tuple[Dataset, Dataset]):
        """Predict the output of the model at points `x` using the linear regressor provided in `state`.

        :param x: the input Dataset `dict` mapping input variables to prediction locations
        :param state: the state containing the linear regressor to use
        :param training_data: not used for `Linear` (since the regressor is already trained in `state`)
        """
        # Convert to (N, xdim) array for sklearn
        x_arr = np.concatenate([x[var][..., np.newaxis] for var in state.x_vars], axis=-1)
        loop_shape = x_arr.shape[:-1]
        x_arr = x_arr.reshape((-1, x_arr.shape[-1]))

        y_arr = state.regressor.predict(x_arr)
        y_arr = y_arr.reshape(loop_shape + (len(state.y_vars),))  # (..., ydim)

        # Unpack the outputs back into a Dataset
        return {var: y_arr[..., i] for i, var in enumerate(state.y_vars)}

    def gradient(self, x: Dataset, state: LinearState, training_data: tuple[Dataset, Dataset]):
        raise NotImplementedError

    def hessian(self, x: Dataset, state: LinearState, training_data: tuple[Dataset, Dataset]):
        raise NotImplementedError


@dataclass
class GPR(Interpolator, StringSerializable):
    """Implementation of Gaussian Process Regression using `sklearn`. The `GPR` uses a pipeline
    of a scaler and a `GaussianProcessRegressor` to approximate the input-output mapping.

    :ivar scaler: the scikit-learn preprocessing scaler to use (e.g. 'MinMaxScaler', 'StandardScaler', etc.). If None,
                  no scaling is applied (default).
    :ivar kernel: the kernel to use for building the covariance matrix (e.g. 'RBF', 'Matern', 'PairwiseKernel', etc.).
                  If a string is provided, then the specified kernel is used with the given `kernel_opts`.
                  If a list is provided, then kernel operators ('Sum', 'Product', 'Exponentiation') can be used to
                  combine multiple kernels. The first element of the list should be the kernel or operator name, and
                  the remaining elements should be the arguments. Dicts are accepted as **kwargs. For example:
                  `['Sum', ['RBF', {'length_scale': 1.0}], ['Matern', {'length_scale': 1.0}]]` will create a sum of
                  an RBF and a Matern kernel with the specified length scales.
    :ivar scaler_opts: options to pass to the scaler constructor
    :ivar kernel_opts: options to pass to the kernel constructor (ignored if kernel is a list, where opts are already
                       specified for combinations of kernels).
    :ivar regressor_opts: options to pass to the `GaussianProcessRegressor` constructor
                          (see [scikit-learn](https://scikit-learn.org/stable/) documentation).
    """
    scaler: str = None
    kernel: str | list = 'RBF'
    scaler_opts: dict = field(default_factory=dict)
    kernel_opts: dict = field(default_factory=dict)
    regressor_opts: dict = field(default_factory=lambda: {'n_restarts_optimizer': 5})

    def _construct_kernel(self, kernel_list):
        """Build a scikit-learn kernel from a list of kernels (e.g. RBF, Matern, etc.) and kernel operators
        (Sum, Product, Exponentiation).

        !!! Example
            `['Sum', ['RBF'], ['Matern', {'length_scale': 1.0}]]` will become `RBF() + Matern(length_scale=1.0)`

        :param kernel_list: list of kernel/operator names and arguments. Kwarg options can be passed as dicts.
        :returns: the scikit-learn kernel object
        """
        # Base case for single items (just return as is)
        if not isinstance(kernel_list, list):
            return kernel_list

        name = kernel_list[0]
        args = [self._construct_kernel(ele) for ele in kernel_list[1:]]

        # Base case for passing a single dict of kwargs
        if len(args) == 1 and isinstance(args[0], dict):
            return getattr(kernels, name)(**args[0])

        # Base case for passing a list of args
        return getattr(kernels, name)(*args)

    def _validate_kernel(self, kernel_list):
        """Make sure all requested kernels are available in scikit-learn."""
        if not isinstance(kernel_list, list):
            return

        name = kernel_list[0]

        if not hasattr(kernels, name):
            raise ImportError(f"Kernel '{name}' not found in sklearn.gaussian_process.kernels")

        for ele in kernel_list[1:]:
            self._validate_kernel(ele)

    def __post_init__(self):
        self._validate_kernel(self.kernel if isinstance(self.kernel, list) else [self.kernel, self.kernel_opts])

        if self.scaler is not None:
            try:
                getattr(preprocessing, self.scaler)
            except AttributeError:
                raise ImportError(f"Scaler '{self.scaler}' not found in sklearn.preprocessing")

    def refine(self, beta: MultiIndex, training_data: tuple[Dataset, Dataset],
               old_state: GPRState, input_domains: dict[str, tuple]) -> InterpolatorState:
        """Train a new gaussian process regression model.

        :param beta: refinement level indices (Not used for 'GPR')
        :param training_data: a tuple of dictionaries containing the new training data (`xtrain`, `ytrain`)
        :param old_state: the old regressor state to refine (only used to get the order of input/output variables)
        :param input_domains: (not used for `GPR`)
        :returns: the new GPR state
        """
        gp_kernel = self._construct_kernel(self.kernel if isinstance(self.kernel, list)
                                           else [self.kernel, self.kernel_opts])
        gp = GaussianProcessRegressor(kernel=gp_kernel, **self.regressor_opts)
        pipe = []
        if self.scaler is not None:
            pipe.append(('scaler', getattr(preprocessing, self.scaler)(**self.scaler_opts)))
        pipe.append(('gpr', gp))
        regressor = Pipeline(pipe)

        xtrain, ytrain = training_data

        # Get order of variables for inputs and outputs
        if old_state is not None:
            x_vars = old_state.x_vars
            y_vars = old_state.y_vars
        else:
            x_vars = list(xtrain.keys())
            y_vars = list(ytrain.keys())

        # Convert to (N, xdim) and (N, ydim) arrays
        x_arr = np.concatenate([xtrain[var][..., np.newaxis] for var in x_vars], axis=-1)
        y_arr = np.concatenate([ytrain[var][..., np.newaxis] for var in y_vars], axis=-1)

        regressor.fit(x_arr, y_arr)

        return GPRState(regressor=regressor, x_vars=x_vars, y_vars=y_vars)

    def predict(self, x: Dataset, state: GPRState, training_data: tuple[Dataset, Dataset]):
        """Predict the output of the model at points `x` using the Gaussian Process Regressor provided in `state`.

        :param x: the input Dataset `dict` mapping input variables to prediction locations
        :param state: the state containing the Gaussian Process Regressor to use
        :param training_data: not used for `GPR` (since the regressor is already trained in `state`)
        """
        # Convert to (N, xdim) array for sklearn
        x_arr = np.concatenate([x[var][..., np.newaxis] for var in state.x_vars], axis=-1)
        loop_shape = x_arr.shape[:-1]
        x_arr = x_arr.reshape((-1, x_arr.shape[-1]))

        y_arr = state.regressor.predict(x_arr)
        y_arr = y_arr.reshape(loop_shape + (len(state.y_vars),))  # (..., ydim)

        # Unpack the outputs back into a Dataset
        return {var: y_arr[..., i] for i, var in enumerate(state.y_vars)}

    def gradient(self, x: Dataset, state: GPRState, training_data: tuple[Dataset, Dataset]):
        raise NotImplementedError

    def hessian(self, x: Dataset, state: GPRState, training_data: tuple[Dataset, Dataset]):
        raise NotImplementedError
