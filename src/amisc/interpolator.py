"""`interpolator.py`

Provides interpolator classes. Interpolators manage training data and specify how to refine/gather more data.

Includes
--------
- `BaseInterpolator`: Abstract class providing basic structure of an interpolator
- `LagrangeInterpolator`: Concrete implementation for tensor-product barycentric Lagrange interpolation
"""
from abc import ABC, abstractmethod
import itertools
import copy

import numpy as np
from scipy.optimize import direct
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from amisc.utils import get_logger
from amisc.rv import BaseRV


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
    :vartype x_vars: list[BaseRV]
    :vartype xi: np.ndarray
    :vartype yi: np.ndarray
    :vartype _model: callable[np.ndarray] -> dict
    :vartype _model_args: tuple
    :vartype _model_kwargs: dict
    :vartype model_cost: float
    :vartype output_files: list[str | Path]
    :vartype logger: logging.Logger
    """

    def __init__(self, beta: tuple, x_vars: BaseRV | list[BaseRV], xi=None, yi=None,
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
        x_vars = [x_vars] if not isinstance(x_vars, list) else x_vars
        self.logger = get_logger(self.__class__.__name__)
        self._model = model
        self._model_args = model_args
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.output_files = []                              # Save output files with same indexing as xi, yi
        self.xi = xi                                        # Interpolation points
        self.yi = yi                                        # Function values at interpolation points
        self.beta = beta                                    # Refinement level indices
        self.x_vars = x_vars                                # BaseRV() objects for each input
        self.model_cost = None                              # Total cpu time to evaluate model once (s)

    def update_input_bds(self, idx: int, bds: tuple):
        """Update the input bounds at the given index.

        :param idx: the index of the input variable to update
        :param bds: the new bounds for the variable
        """
        self.x_vars[idx].update_bounds(*bds)

    def xdim(self):
        """Get the dimension of the input domain."""
        return len(self.x_vars)

    def ydim(self):
        """Get the dimension of the outputs."""
        return self.yi.shape[-1] if self.yi is not None else None

    def save_enabled(self):
        """Return whether the underlying model wants to save outputs to file.

        !!! Note
            You can specify that a model wants to save outputs to file by providing an `'output_dir'` kwarg.
        """
        return self._model_kwargs.get('output_dir') is not None

    def set_yi(self, yi: np.ndarray = None, model: callable = None,
               x_new: tuple[list[int | tuple], np.ndarray] = ()) -> dict[str: np.ndarray] | None:
        """Set the training data; if `yi=None`, then compute from the model.

        !!! Warning
            You would use `x_new` if you wanted to compute the model at these specific locations and store the result.
            This will ignore anything passed in for `yi`, and it assumes a model is already specified (or passed in).

        !!! Info
            You can pass in integer indices for `x_new` or tuple indices. Integers will index into `self.xi`. Tuples
            provide extra flexibility for more complicated indexing, e.g. they might specify indices along different
            coordinate directions in an N-dimensional grid. If you pass in a list of tuple indices for `x_new`, the
            resulting model outputs will be returned back to you in the form `dict[str: np.ndarray]`. The keys are
            string casts of the tuple indices, and the values are the corresponding model outputs.

        :param yi: `(Nx, y_dim)`, training data to set, must match dimension of `self.xi`
        :param model: callable function, optionally overrides `self._model`
        :param x_new: tuple of `(idx, x)`, where `x` is an `(N_new, x_dim)` array of new interpolation points to
                      include and `idx` specifies the indices of these points into `self.xi`
        :returns: dict[str: np.ndarray] if `idx` contains tuple elements, otherwise `None`
        """
        if model is not None:
            self._model = model
        if self._model is None:
            error_msg = 'Model not specified for computing QoIs at interpolation grid points.'
            self.logger.error(error_msg)
            raise Exception(error_msg)

        # Overrides anything passed in for yi (you would only be using this if yi was set previously)
        if x_new:
            new_idx = x_new[0]
            new_x = x_new[1]
            return_y = isinstance(new_idx[0], tuple)  # Return y rather than storing it if tuple indices are passed in
            ret = dict(y=dict(), files=dict())
            model_ret = self._model(new_x, *self._model_args, **self._model_kwargs)
            if not isinstance(model_ret, dict):
                self.logger.warning(
                    f"Function {self._model} did not return a dict of the form {{'y': y}}. Please make sure"
                    f" you do so to avoid conflicts. Returning the value directly instead...")
                model_ret = dict(y=model_ret)
            y_new, files_new, cpu_time = model_ret['y'], model_ret.get('files', None), model_ret.get('cost', 1)

            if self.save_enabled():
                for j in range(y_new.shape[0]):
                    if return_y:
                        ret['y'][str(new_idx[j])] = y_new[j, :].astype(np.float32)
                        ret['files'][str(new_idx[j])] = files_new[j]
                    else:
                        self.yi[new_idx[j], :] = y_new[j, :].astype(np.float32)
                        self.output_files[new_idx[j]] = files_new[j]
            else:
                for j in range(y_new.shape[0]):
                    if return_y:
                        ret['y'][str(new_idx[j])] = y_new[j, :].astype(np.float32)
                    else:
                        self.yi[new_idx[j], :] = y_new[j, :].astype(np.float32)

            if self.model_cost is None:
                self.model_cost = max(1, cpu_time)

            return ret

        # Set yi directly
        if yi is not None:
            self.yi = yi.astype(np.float32)
            return

        # Compute yi
        model_ret = self._model(self.xi, *self._model_args, **self._model_kwargs)
        if not isinstance(model_ret, dict):
            self.logger.warning(f"Function {self._model} did not return a dict of the form {{'y': y}}. Please make sure"
                                f" you do so to avoid conflicts. Returning the value directly instead...")
            model_ret = dict(y=model_ret)

        self.yi, self.output_files, cpu_time = model_ret['y'], model_ret.get('files', list()), model_ret.get('cost', 1)

        if self.model_cost is None:
            self.model_cost = max(1, cpu_time)

    @abstractmethod
    def refine(self, beta: tuple, auto=True):
        """Return a new interpolator with one dimension refined by one level, as specified by `beta`.

        !!! Info "When you want to compute the model manually"
            You can set `auto=False`, in which case the newly refined interpolation points `x` will be returned to you
            along with their indices, in the form `idx, x, interp = refine(beta, auto=False)`. You might also want to
            do this if you did not provide a model when constructing the Interpolator (so `auto=True` won't work).

        :param beta: the new refinement level indices, should only refine one dimension by one level
        :param auto: whether to automatically compute and store model at refinement points (default is True)
        :returns: `idx` - indices into `xi`, `x` - the new interpolation points, and `interp` - a refined
                   BaseInterpolator object, just returns `interp` if `auto=True`
        """
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate the interpolation at points `x`.

        :param x: `(..., x_dim)`, the points to be interpolated, must be within the input domain for accuracy
        :returns y: `(..., y_dim)`, the interpolated function values
        """
        pass


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

    def __init__(self, beta: tuple, x_vars: BaseRV | list[BaseRV], init_grids=True, reduced=False, **kwargs):
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
            self.x_grids = [self.leja_1d(grid_sizes[n], self.x_vars[n].bounds(),
                                         wt_fcn=self.x_vars[n].pdf).astype(np.float32) for n in range(self.xdim())]

            for n in range(self.xdim()):
                Nx = grid_sizes[n]
                bds = self.x_vars[n].bounds()
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
                                                                                   interp.x_vars[dim_refine].bounds(),
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
            bds = interp.x_vars[dim_refine].bounds()
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

            old_indices = [np.arange(old_grid_sizes[n]) for n in range(self.xdim())]
            old_indices = list(itertools.product(*old_indices))
            new_indices = [np.arange(new_grid_sizes[n]) for n in range(self.xdim())]
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
                self.logger.warning(f'No model available to evaluate new interpolation points, returning the points '
                                    f'to you instead...')
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
        x = np.atleast_1d(x)
        if yi is None:
            yi = self.yi.copy()
        if xi is None:
            xi = self.xi.copy()
        ydim = yi.shape[-1]
        nan_idx = np.any(np.isnan(yi), axis=-1)
        if np.any(nan_idx):
            # Use a simple linear regression fit to impute missing values (may have resulted from bad model outputs)
            imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
            imputer.fit(xi[~nan_idx, :], yi[~nan_idx, :])
            yi[nan_idx, :] = imputer.predict(xi[nan_idx, :])

        # Loop over multi-indices and compute tensor-product lagrange polynomials
        grid_sizes = self.get_grid_sizes(self.beta)
        y = np.zeros(x.shape[:-1] + (ydim,))    # (..., ydim)
        indices = [np.arange(grid_sizes[n]) for n in range(self.xdim())]
        for i, j in enumerate(itertools.product(*indices)):
            L_j = np.empty(x.shape)             # (..., xdim)

            # Compute univariate Lagrange polynomials in each dimension
            for n in range(self.xdim()):
                x_n = x[..., n, np.newaxis]     # (..., 1)
                x_j = self.x_grids[n]           # (Nx,)
                w_j = self.weights[n]           # (Nx,)

                # Compute the jth Lagrange basis polynomial L_j(x_n) for this x dimension (in barycentric form)
                c = x_n - x_j
                div_zero_idx = np.abs(c) <= 1e-4 * np.abs(x_j) + 1e-8   # Track where x is at an interpolation pnt x_j
                c[div_zero_idx] = 1                             # Temporarily set to 1 to avoid divide by zero error
                c = w_j / c
                L_j[..., n] = c[..., j[n]] / np.sum(c, axis=-1)  # (...) same size as original x

                # Set L_j(x==x_j)=1 for the current j and set L_j(x==x_j)=0 for x_j = x_i, i != j
                L_j[div_zero_idx[..., j[n]], n] = 1
                L_j[np.any(div_zero_idx[..., [idx for idx in range(grid_sizes[n]) if idx != j[n]]], axis=-1), n] = 0

            # Add multivariate basis polynomial contribution to interpolation output
            L_j = np.prod(L_j, axis=-1, keepdims=True)      # (..., 1)
            y += L_j * yi[i, :]

        return y

    @staticmethod
    def get_grid_sizes(beta: tuple, k: int = 2) -> list[int]:
        """Compute number of grid points in each input dimension.

        :param beta: refinement level indices
        :param k: level-to-grid-size multiplier (probably just always `k=2`)
        :returns: list of grid sizes in each dimension
        """
        return [k*beta[i] + 1 for i in range(len(beta))]

    @staticmethod
    def leja_1d(N: int, z_bds: tuple, z_pts: np.ndarray = None, wt_fcn: callable = None) -> np.ndarray:
        """Find the next `N` points in the Leja sequence of `z_pts`.

        :param N: number of new points to add to the sequence
        :param z_bds: bounds on the 1d domain
        :param z_pts: current univariate Leja sequence `(Nz,)`, start at middle of `z_bds` if `None`
        :param wt_fcn: weighting function, uses a constant weight if `None`, callable as `wt_fcn(z)`
        :returns: the Leja sequence `z_pts` augmented by `N` new points
        """
        # if wt_fcn is None:
        wt_fcn = lambda z: 1  # UPDATE: ignore RV weighting, unbounded pdfs like Gaussian cause problems
        if z_pts is None:
            z_pts = (z_bds[1] + z_bds[0]) / 2
            N = N - 1
        z_pts = np.atleast_1d(z_pts).astype(np.float32)

        # Construct Leja sequence by maximizing the Leja objective sequentially
        for i in range(N):
            obj_fun = lambda z: -wt_fcn(np.array(z).astype(np.float32)) * np.prod(np.abs(z - z_pts))
            res = direct(obj_fun, [z_bds])  # Use global DIRECT optimization over 1d domain
            z_star = res.x
            z_pts = np.concatenate((z_pts, z_star))

        return z_pts
