"""Provides an object-oriented interface for model inputs/outputs and random variables.

Includes:

- `Distribution` — an object for specifying a PDF. `Normal`, `Uniform`, `Relative`, and `Tolerance` are available.
- `Transform` — an object for specifying a transformation. `Linear`, `Log`, `Minmax`, and `Zscore` are available.
- `Compression` — an object for specifying a compression method for field quantities. `SVD` is available.
- `CompressionData` — a dictionary spec for passing data to/from `Variable.compress()`
- `Variable` — an object that stores information about a variable and includes methods for sampling, pdf evaluation,
               normalization, compression, loading from file, etc.
- `VariableList` — a container for Variables that provides dict-like access of Variables by `name` along with normal
                   indexing and slicing

The preferred serialization of `Variable` and `VariableList` is to/from yaml. `Distribution` and `Transform`
objects can be serialized via conversion to/from string. `Compression` objects can be serialized via pickle.
"""
from __future__ import annotations

import ast
import inspect
import random
import string
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Union

import numpy as np
import yaml
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from scipy.interpolate import RBFInterpolator
from typing_extensions import TypedDict

from amisc.serialize import Serializable, PickleSerializable
from amisc.utils import parse_function_string, search_for_file, _get_yaml_path, _inspect_assignment

__all__ = ['Distribution', 'Uniform', 'Normal', 'Relative', 'Tolerance', 'CompressionData', 'Compression', 'SVD',
           'Transform', 'Linear', 'Log', 'Minmax', 'Zscore', 'Variable', 'VariableList']


class Distribution(ABC):
    """Base class for PDF distributions that provide sample and pdf methods."""

    def __init__(self, dist_args: tuple):
        self.dist_args = dist_args

    def __str__(self):
        """Serialize a `Distribution` object to/from string."""
        return f'{type(self).__name__}{self.dist_args}'

    def __repr__(self):
        return self.__str__()

    def domain(self, dist_args: tuple = None) -> tuple:
        """Return the domain of this distribution. Defaults to `dist_args`

        :param dist_args: overrides `self.dist_args`
        """
        return dist_args or self.dist_args

    def nominal(self, dist_args: tuple = None) -> float:
        """Return the nominal value of this distribution. Defaults to middle of domain.

        :param dist_args: overrides `self.dist_args`
        """
        lb, ub = self.domain(dist_args=dist_args)
        return (lb + ub) / 2

    @classmethod
    def from_string(cls, dist_string: str) -> Distribution | None:
        """Convert a string to a `Distribution` object.

        :param dist_string: specifies a PDF or distribution. Can be `Normal(mu, std)`, `Uniform(lb, ub)`,
                            `Relative(pct)`, or `Tolerance(tol)`. The shorthands `N(0, 1)`, `U(0, 1)`, `rel(5)`, or
                            `tol(1)` are also accepted.
        :return: the corresponding `Distribution` object
        """
        if not dist_string:
            return None

        dist_name, args, kwargs = parse_function_string(dist_string)
        if dist_name in ['N', 'Normal', 'normal']:
            # Normal distribution like N(0, 1)
            try:
                mu = float(kwargs.get('mu', args[0]))
                std = float(kwargs.get('std', args[1]))
                return Normal((mu, std))
            except Exception as e:
                raise ValueError(f'Normal distribution string "{dist_string}" is not valid: Try N(0, 1).') from e
        elif dist_name in ['U', 'Uniform', 'uniform']:
            # Uniform distribution like U(0, 1)
            try:
                lb = float(kwargs.get('lb', args[0]))
                ub = float(kwargs.get('ub', args[1]))
                return Uniform((lb, ub))
            except Exception as e:
                raise ValueError(f'Uniform distribution string "{dist_string}" is not valid: Try U(0, 1).') from e
        elif dist_name in ['R', 'Relative', 'relative', 'rel']:
            # Relative uniform distribution like rel(+-5%)
            try:
                pct = float(kwargs.get('pct', args[0]))
                return Relative((pct,))
            except Exception as e:
                raise ValueError(f'Relative distribution string "{dist_string}" is not valid: Try rel(5).') from e
        elif dist_name in ['T', 'Tolerance', 'tolerance', 'tol']:
            # Uniform distribution within a tolerance like tol(+-1)
            try:
                tol = float(kwargs.get('tol', args[0]))
                return Tolerance((tol,))
            except Exception as e:
                raise ValueError(f'Tolerance distribution string "{dist_string}" is not valid: Try tol(1).') from e
        else:
            raise NotImplementedError(f'The distribution "{dist_string}" is not recognized.')

    @abstractmethod
    def sample(self, shape: int | tuple, nominal: float | np.ndarray = None, dist_args: tuple = None) -> np.ndarray:
        """Sample from the distribution.

        :param shape: shape of the samples to return
        :param nominal: a nominal value(s) for sampling (e.g. for relative distributions)
        :param dist_args: overrides `Distribution.dist_args`
        :return: the samples of the given shape
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: np.ndarray, dist_args: tuple = None) -> np.ndarray:
        """Evaluate the pdf of this distribution at the `x` locations.

        :param x: the locations at which to evaluate the pdf
        :param dist_args: overrides `Distribution.dist_args`
        :return: the pdf evaluations
        """
        raise NotImplementedError


class Uniform(Distribution):
    """A Uniform distribution. Specify by string as "Uniform(lb, ub)" or "U(lb, ub)" in shorthand."""

    def __str__(self):
        return f'U({self.dist_args[0]}, {self.dist_args[1]})'

    def sample(self, shape, nominal=None, dist_args=None):
        lb, ub = dist_args or self.dist_args
        return np.random.rand(*shape) * (ub - lb) + lb

    def pdf(self, x, dist_args=None):
        lb, ub = dist_args or self.dist_args
        pdf = np.broadcast_to(1 / (ub - lb), x.shape).copy()
        pdf[np.where(x > ub)] = 0
        pdf[np.where(x < lb)] = 0
        return pdf


class Normal(Distribution):
    """A Normal distribution. Specify by string as "Normal(mu, std)" or "N(mu, std)" in shorthand."""

    def __str__(self):
        return f'N({self.dist_args[0]}, {self.dist_args[1]})'

    def domain(self, dist_args=None):
        mu, std = dist_args or self.dist_args
        return mu - 3 * std, mu + 3 * std

    def sample(self, shape, nominal=None, dist_args=None):
        mu, std = dist_args or self.dist_args
        return np.random.randn(*shape) * std + mu

    def pdf(self, x, dist_args=None):
        mu, std = dist_args or self.dist_args
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) / std) ** 2)


class Relative(Distribution):
    """A Relative distribution. Specify by string as "Relative(pct)" or "rel(pct%)" in shorthand.
    Will attempt to sample uniformly within the given percent of a nominal value.
    """

    def __str__(self):
        return rf'rel({self.dist_args[0]})'

    def domain(self, dist_args=None):
        return None

    def nominal(self, dist_args=None):
        return None

    def sample(self, shape, nominal=None, dist_args=None):
        if nominal is None:
            raise ValueError('Cannot sample relative distribution when no nominal value is provided.')
        dist_args = dist_args or self.dist_args
        tol = abs((dist_args[0] / 100) * nominal)
        return np.random.rand(*shape) * 2 * tol - tol + nominal

    def pdf(self, x, dist_args=None):
        return np.ones(x.shape)


class Tolerance(Distribution):
    """A Tolerance distribution. Specify by string as "Tolerance(tol)" or "tol(tol)" in shorthand.
    Will attempt to sample uniformly within a given absolute tolerance of a nominal value.
    """

    def __str__(self):
        return rf'tol({self.dist_args[0]})'

    def domain(self, dist_args=None):
        return None

    def nominal(self, dist_args=None):
        return None

    def sample(self, shape, nominal=None, dist_args=None):
        if nominal is None:
            raise ValueError('Cannot sample tolerance distribution when no nominal value is provided.')
        dist_args = dist_args or self.dist_args
        tol = abs(dist_args[0])
        return np.random.rand(*shape) * 2 * tol - tol + nominal

    def pdf(self, x, dist_args=None):
        return np.ones(x.shape)


@dataclass
class Compression(PickleSerializable, ABC):
    """Base class for compression methods."""
    fields: list[str] = field(default_factory=list)
    method: str = 'svd'
    shape: tuple = ()
    coords: np.ndarray = None  # (num_pts, dim)
    interpolate_method: str = 'rbf'
    interpolate_opts: dict = field(default_factory=dict)
    _map_computed: bool = False

    @property
    def map_exists(self):
        """All compression methods should have `coords` when their map has been constructed."""
        return self.coords is not None and self._map_computed

    @property
    def dim(self):
        """Number of physical grid coordinates for the field quantity, (i.e. x,y,z spatial dims)"""
        return self.coords.shape[1] if (self.coords is not None and len(self.coords.shape) > 1) else 1

    @property
    def num_pts(self):
        """Number of physical points in the compression grid"""
        return self.coords.shape[0] if self.coords is not None else None

    @property
    def num_qoi(self):
        """Number of quantities of interest at each grid point, (i.e. ux, uy, uz for 3d velocity data)"""
        return len(self.fields) if self.fields is not None else 1

    @property
    def dof(self):
        """Total degrees of freedom in the compression grid."""
        return self.num_pts * self.num_qoi if self.num_pts is not None else None

    def _correct_coords(self, coords):
        """Correct the coordinates to be in the correct shape for compression."""
        coords = np.atleast_1d(coords)
        if len(coords.shape) == 1:
            coords = coords[..., np.newaxis] if self.dim == 1 else coords[np.newaxis, ...]
        return coords

    def interpolator(self):
        """The interpolator to use during compression and reconstruction. Interpolator expects to be used as:

        ```python
        xg = np.ndarray    # (num_pts, dim)  grid coordinates
        yg = np.ndarray    # (num_pts, ...)  scalar values on grid
        xp = np.ndarray    # (Q, dim)        evaluation points

        interp = interpolate_method(xg, yg, **interpolate_opts)

        yp = interp(xp)    # (Q, ...)        interpolated values
        ```
        """
        method = self.interpolate_method or 'rbf'
        match method.lower():
            case 'rbf':
                return RBFInterpolator
            case other:
                raise NotImplementedError(f"Interpolation method '{other}' is not implemented.")

    def interpolate_from_grid(self, states, new_coords):
        """Interpolate the states on the compression grid to new coordinates.

        :param states: `(*loop_shape, dof)` - the states on the compression grid
        :param new_coords: `(*coord_shape, dim)` - the new coordinates to interpolate to
        :return: `dict` of `(*loop_shape, *coord_shape)` for each qoi - the interpolated states
        """
        new_coords = self._correct_coords(new_coords)
        grid_coords = self._correct_coords(self.coords)
        skip_interp = (new_coords.shape == grid_coords.shape and np.allclose(new_coords, grid_coords))

        ret_dict = {}
        loop_shape = states.shape[:-1]
        coords_shape = new_coords.shape[:-1]
        states = states.reshape((*loop_shape, self.num_pts, self.num_qoi))
        new_coords = new_coords.reshape((-1, self.dim))
        for i, qoi in enumerate(self.fields):
            if skip_interp:
                ret_dict[qoi] = states[..., i]
            else:
                reshaped_states = states[..., i].reshape(-1, self.num_pts).T  # (num_pts, ...)
                interp = self.interpolator()(grid_coords, reshaped_states, **self.interpolate_opts)
                yp = interp(new_coords)
                ret_dict[qoi] = yp.T.reshape(*loop_shape, *coords_shape)

        return ret_dict

    def interpolate_to_grid(self, field_coords, field_values):
        """Interpolate the field values at given coordinates to the compression grid.

        :param field_coords: `(*coord_shape, dim)` - the coordinates of the field values
        :param field_values: `dict` of `(*loop_shape, *coord_shape)` for each qoi - the field values at the coordinates
        :return: `(*loop_shape, dof)` - the interpolated values on the compression grid
        """
        field_coords = self._correct_coords(field_coords)
        grid_coords = self._correct_coords(self.coords)
        skip_interp = (field_coords.shape == grid_coords.shape and np.allclose(field_coords, grid_coords))

        coords_shape = field_coords.shape[:-1]
        loop_shape = next(iter(field_values.values())).shape[:-len(coords_shape)]
        states = np.empty((*loop_shape, self.num_pts, self.num_qoi))
        field_coords = field_coords.reshape(-1, self.dim)
        for i, qoi in enumerate(self.fields):
            field_vals = field_values[qoi].reshape((*loop_shape, -1))  # (..., Q)
            if skip_interp:
                states[..., i] = field_vals
            else:
                field_vals = field_vals.reshape((-1, field_vals.shape[-1])).T  # (Q, ...)
                interp = self.interpolator()(field_coords, field_vals, **self.interpolate_opts)
                yg = interp(grid_coords)
                states[..., i] = yg.T.reshape(*loop_shape, self.num_pts)

        return states.reshape((*loop_shape, self.dof))

    @abstractmethod
    def compute_map(self, **kwargs):
        """Compute and store the compression map. Must set the value of `coords` and `_is_computed`."""
        raise NotImplementedError

    @abstractmethod
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress the data into a latent space.

        :param data: `(..., dof)` - the data to compress from full size of `dof`
        :return: `(..., rank)` - the compressed latent space data
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        """Reconstruct the compressed data back into the full `dof` space.

        :param compressed: `(..., rank)` - the compressed data to reconstruct
        :return: `(..., dof)` - the reconstructed data with full `dof`
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, spec: dict) -> Compression:
        """Construct a `Compression` object from a spec dictionary."""
        method = spec.pop('method', 'svd').lower()
        match method:
            case 'svd':
                return SVD(**spec)
            case other:
                raise NotImplementedError(f"Compression method '{other}' is not implemented.")


@dataclass
class SVD(Compression):
    """A Singular Value Decomposition (SVD) compression method."""
    data_matrix: np.ndarray = None          # (dof, num_samples)
    projection_matrix: np.ndarray = None    # (dof, rank)
    rank: int = None
    energy_tol: float = None

    def __post_init__(self):
        if (data_matrix := self.data_matrix) is not None:
            self.compute_map(data_matrix, rank=self.rank, energy_tol=self.energy_tol)

    def compute_map(self, data_matrix: np.ndarray, rank: int = None, energy_tol: float = None):
        """Compute the SVD compression map from the data matrix.

        :param data_matrix: `(dof, num_samples)` - the data matrix
        :param rank: the rank of the SVD decomposition
        :param energy_tol: the energy tolerance of the SVD decomposition
        """
        nan_idx = np.any(np.isnan(data_matrix), axis=0)
        data_matrix = data_matrix[:, ~nan_idx]
        u, s, vt = np.linalg.svd(data_matrix)
        energy_frac = np.cumsum(s ** 2 / np.sum(s ** 2))
        if rank := (rank or self.rank):
            energy_tol = energy_frac[rank - 1]
        else:
            energy_tol = energy_tol or self.energy_tol or 0.95
            idx = int(np.where(energy_frac >= energy_tol)[0][0])
            rank = idx + 1

        self.rank = rank
        self.energy_tol = energy_tol
        self.projection_matrix = u[:, :rank]  # (dof, rank)
        self._map_computed = True

    def compress(self, data):
        return np.squeeze(self.projection_matrix.T @ data[..., np.newaxis], axis=-1)

    def reconstruct(self, compressed):
        return np.squeeze(self.projection_matrix @ compressed[..., np.newaxis], axis=-1)


class CompressionData(TypedDict, total=False):
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
    coord: np.ndarray
    latent: np.ndarray
    qty: np.ndarray


class Transform(ABC):
    """A base class for all transformations.

    :ivar transform_args: the arguments for the transformation
    :vartype transform_args: tuple
    """

    def __init__(self, transform_args: tuple):
        self.transform_args = transform_args

    def __str__(self):
        """Serialize a `Transform` object to string."""
        return f'{type(self).__name__}{self.transform_args}'

    @classmethod
    def from_string(cls, transform_spec: str | list[str]) -> list[Transform] | None:
        """Return a list of Transforms given a list of string specifications. Available transformations are:

        - **linear** — $x_{norm} = mx + b$ specified as `linear(m, b)` or `linear(slope=m, offset=b)`. `m=1, b=0` if not
                       specified.
        - **log** — $x_{norm} = \\log_b(x)$ specified as `log` or `log10` for the natural or common logarithms. For a
                    different base, use `log(b)`. Optionally, specify `offset` for `log(x+offset)`.
        - **minmax** — $x_{norm} = \\frac{x - a}{b - a}(u - l) + l$ specified as `minmax(a, b, l, u)` or
                       `minmax(lb=a, ub=b, lb_norm=l, ub_norm=u)`. Scales `x` from the range `(a, b)` to `(l, u)`. By
                       default, `(a, b)` is the Variable's domain and `(l, u)` is `(0, 1)`. Use simply as `minmax`
                       to use all defaults.
        - **zscore** — $x_{norm} = \\frac{x - m}{s}$ specified as `zscore(m, s)` or `zscore(mu=m, std=s)`. If the
                       Variable is specified as `dist=normal`, then `zscore` defaults to the Variable's own `mu, std`.

        !!! Example
            ```python
            transforms = Transform.from_string(['log10', 'linear(2, 4)'])
            print(transforms)
            ```
            will give
            ```shell
            [ Log(10), Linear(2, 4) ]  # the corresponding `Transform` objects
            ```

        !!! Warning
            You may optionally leave the `minmax` arguments blank to defer to the bounds of the parent `Variable`.
            You may also optionally leave the `zscore` arguments blank to defer to the `(mu, std)` of the parent
            `Variable`, but this will throw a runtime error if `Variable.dist` is not `Normal(mu, std)`.
        """
        if transform_spec is None:
            return None
        if isinstance(transform_spec, str | Transform):
            transform_spec = [transform_spec]

        transforms = []
        for spec_string in transform_spec:
            if isinstance(spec_string, Transform):
                transforms.append(spec_string)
                continue

            name, args, kwargs = parse_function_string(spec_string)
            if name.lower() == 'linear':
                try:
                    slope = float(kwargs.get('slope', args[0] if len(args) > 0 else 1))
                    offset = float(kwargs.get('offset', args[1] if len(args) > 1 else 0))
                    transforms.append(Linear((slope, offset)))
                except Exception as e:
                    raise ValueError(f'Linear transform spec "{spec_string}" is not valid: Try "linear(m, b)".') from e
            elif name.lower() in ['log', 'log10']:
                try:
                    log_base = float(kwargs.get('base', args[0] if len(args) > 0 else (np.e if name.lower() == 'log'
                                                                                       else 10)))
                    offset = float(kwargs.get('offset', args[1] if len(args) > 1 else 0))
                    transforms.append(Log((log_base, offset)))
                except Exception as e:
                    raise ValueError(f'Log transform spec "{spec_string}" is not valid: Try "log(base, offset)"') from e
            elif name.lower() in ['minmax', 'maxabs']:
                try:
                    # Defer bounds to the Variable by setting np.nan
                    lb = float(kwargs.get('lb', args[0] if len(args) > 0 else np.nan))
                    ub = float(kwargs.get('ub', args[1] if len(args) > 1 else np.nan))
                    lb_norm = float(kwargs.get('lb_norm', args[2] if len(args) > 2 else 0))
                    ub_norm = float(kwargs.get('ub_norm', args[3] if len(args) > 3 else 1))
                    transforms.append(Minmax((lb, ub, lb_norm, ub_norm)))
                except Exception as e:
                    raise ValueError(f'Minmax transform spec "{spec_string}" is not valid: Try "minmax(lb, ub)"') from e
            elif name.lower() in ['z', 'zscore']:
                try:
                    # Defer (mu, std) to the Variable by setting np.nan
                    mu = float(kwargs.get('mu', args[0] if len(args) > 0 else np.nan))
                    std = float(kwargs.get('std', args[1] if len(args) > 1 else np.nan))
                    transforms.append(Zscore((mu, std)))
                except Exception as e:
                    raise ValueError(f'Z-score normalization string "{spec_string}" is not valid: '
                                     f'Try "zscore(mu, std)".') from e
            else:
                raise NotImplementedError(f'Transform method "{name}" is not implemented.')

        return transforms

    def transform(self, x: ArrayLike, inverse: bool = False, transform_args: tuple = None) -> ArrayLike:
        """Transform the given values `x`.

        :param x: the values to transform
        :param inverse: whether to do the inverse transform instead
        :param transform_args: overrides `Transform.transform_args`
        :return: the transformed values
        """
        input_type = type(x)
        result = self._transform(np.atleast_1d(x), inverse, transform_args)
        if input_type in [int, float]:
            return float(result[0])
        elif input_type is list:
            return result.tolist()
        elif input_type is tuple:
            return tuple(result.tolist())
        else:
            return result  # just keep as np.ndarray for everything else

    @abstractmethod
    def _transform(self, x, inverse=False, transform_args=None):
        """Abstract method that subclass `Transform` objects should implement."""
        raise NotImplementedError


class Linear(Transform):
    """A Linear transform: $y=mx+b$.

    :ivar transform_args: `(m, b)` the slope and offset
    """
    def _transform(self, x, inverse=False, transform_args=None):
        slope, offset = transform_args or self.transform_args
        return (x - offset) / slope if inverse else slope * x + offset


class Log(Transform):
    """A Log transform: $y=\\log_b{x + offset}$.

    :ivar transform_args: `(base, offset)` the log base and offset
    """
    def _transform(self, x, inverse=False, transform_args=None):
        log_base, offset = transform_args or self.transform_args
        return np.exp(x * np.log(log_base)) - offset if inverse else np.log(x + offset) / np.log(log_base)


class Minmax(Transform):
    """A Minmax transform: $x: (lb, ub) \\mapsto (lb_{norm}, ub_{norm})$.

    :ivar transform_args: `(lb, ub, lb_norm, ub_norm)` the original lower and upper bounds and the normalized bounds
    """
    def _transform(self, x, inverse=False, transform_args=None):
        transform_args = transform_args or self.transform_args
        if np.any(np.isnan(transform_args)):
            raise RuntimeError(f'Transform args may have missing values: {transform_args}')
        lb, ub, lb_norm, ub_norm = transform_args
        if inverse:
            return (x - lb_norm) / (ub_norm - lb_norm) * (ub - lb) + lb
        else:
            return (x - lb) / (ub - lb) * (ub_norm - lb_norm) + lb_norm

    def update(self, lb=None, ub=None, lb_norm=None, ub_norm=None):
        """Update the parameters of this transform.

        :param lb: the lower bound in the original variable space
        :param ub: the upper bound in the original variable space
        :param lb_norm: the lower bound of the transformed space
        :param ub_norm: the upper bound of the transformed space
        """
        transform_args = (lb, ub, lb_norm, ub_norm)
        self.transform_args = tuple([ele if ele is not None else self.transform_args[i]
                                     for i, ele in enumerate(transform_args)])


class Zscore(Transform):
    """A Zscore transform: $y=(x-\\mu)/\\sigma$.

    :ivar transform_args: `(mu, std)` the mean and standard deviation
    """
    def _transform(self, x, inverse=False, transform_args=None):
        transform_args = transform_args or self.transform_args
        if np.any(np.isnan(transform_args)):
            raise RuntimeError(f'Transform args may have missing values: {transform_args}')
        mu, std = transform_args
        return x * std + mu if inverse else (x - mu) / std

    def update(self, mu=None, std=None):
        """Update the parameters of this transform.

        :param mu: the mean of the transform
        :param std: the standard deviation of the transform
        """
        transform_args = (mu, std)
        self.transform_args = tuple([ele if ele is not None else self.transform_args[i]
                                     for i, ele in enumerate(transform_args)])


Transformation = Union[str, Transform, list[str | Transform]]


class Variable(BaseModel, Serializable):
    """Object for storing information about variables and providing methods for pdf evaluation, sampling, etc.

    A simple variable object can be created with `var = Variable()`. All initialization options are optional and will
    be given good defaults. You should probably at the very least give a memorable `name` and a `domain`.

    With the `pyyaml` library installed, all Variable objects can be saved or loaded directly from a `.yml` file by
    using the `!Variable` yaml tag (which is loaded by default with `amisc`).

    - Use `Variable.dist` to specify sampling PDFs, such as for random variables. See the `Distribution` classes.
    - Use `Variable.norm` to specify a transformed-space that is more amenable to surrogate construction
      (e.g. mapping to the range (0,1)). See the `Transform` classes.
    - Use `Variable.compression` to specify high-dimensional, coordinate-based field quantities,
      such as from the output of many simulation software programs. See [`Compression`][amisc.variable.Compression].
    - Use `Variable.category` as an additional layer for using Variable's in different ways (e.g. set a "calibration"
      category for Bayesian inference).

    !!! Example
        ```python
        # Random variable
        temp = Variable(name='T', description='Temperature', units='K', dist='Uniform(280, 320)')
        samples = temp.sample(100)
        pdf = temp.pdf(samples)

        # Field quantity
        vel = Variable(name='u', description='Velocity', units='m/s', field={'quantities': ['ux', 'uy', 'uz']})
        vel_data = ...  # from a simulation
        reduced_vel = vel.compress(vel_data)
        ```

    !!! Warning
        Changes to collection fields (like `Variable.norm`) should completely reassign the _whole_
        collection to trigger the correct validation, rather than editing particular entries. For example, reassign
        `norm=['log', 'linear(2, 2)']` rather than editing norm via `norm.append('linear(2, 2)')`.

    :ivar name: an identifier for the variable, can compare variables directly with strings for indexing purposes
    :ivar nominal: a typical value for this variable
    :ivar description: a lengthier description of the variable
    :ivar units: assumed units for the variable (if applicable)
    :ivar category: an additional descriptor for how this variable is used, e.g. calibration, operating, design, etc.
    :ivar tex: latex format for the variable, i.e. r"$x_i$"
    :ivar compression: specifies field quantities and links to relevant compression data
    :ivar dist: a string specifier of a probability distribution function (or a `Distribution` object)
    :ivar domain: the explicit domain bounds of the variable (limits of where you expect to use it)
    :ivar norm: specifier of a map to a transformed-space for surrogate construction (or a `Transformation` type)
    """
    yaml_tag: ClassVar[str] = u'!Variable'
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, validate_default=True)

    name: Optional[str] = None
    nominal: Optional[float] = None
    description: Optional[str] = None
    units: Optional[str] = None
    category: Optional[str] = None
    tex: Optional[str] = None
    compression: Optional[str | dict | Compression] = None
    dist: Optional[str | Distribution] = None
    domain: Optional[str | tuple[float, float]] = None
    norm: Optional[Transformation] = None

    def __init__(self, /, name=None, **kwargs):
        # Try to set the variable name if instantiated as "x = Variable()"
        if name is None:
            name = _inspect_assignment('Variable')
        name = name or "X_" + "".join(random.choices(string.digits, k=3))
        super().__init__(name=name, **kwargs)

    @field_validator('tex')
    @classmethod
    def _validate_tex(cls, tex: str) -> str | None:
        if tex is None:
            return tex
        if not tex.startswith('$'):
            tex = rf'${tex}'
        if not tex[-1] == '$':
            tex = rf'{tex}$'
        return tex

    @field_validator('compression')
    @classmethod
    def _validate_compression(cls, compression: str | dict | Compression, info: ValidationInfo) -> Compression | None:
        if compression is None:
            return compression
        elif isinstance(compression, str):
            return Compression.deserialize(compression)
        elif isinstance(compression, dict):
            compression['fields'] = compression.get('fields', None) or [info.data['name']]
            return Compression.from_dict(compression)
        else:
            compression.fields = compression.fields or [info.data['name']]
            return compression

    @field_validator('dist')
    @classmethod
    def _validate_dist(cls, dist: str | Distribution) -> Distribution | None:
        if dist is None:
            return dist
        if isinstance(dist, Distribution):
            return dist
        elif isinstance(dist, str):
            return Distribution.from_string(dist)
        else:
            raise ValueError(f'Cannot convert {dist} to a Distribution object.')

    @field_validator('domain')
    @classmethod
    def _validate_domain(cls, domain: tuple | str, info: ValidationInfo) -> tuple | None:
        """Try to extract the domain from the distribution if not provided, or convert from a string."""
        if domain is None:
            if dist := info.data['dist']:
                domain = dist.domain()
        elif isinstance(domain, str):
            domain = tuple(ast.literal_eval(domain.strip()))
        if domain is None:
            return domain
        assert isinstance(domain, tuple) and len(domain) == 2
        assert domain[1] > domain[0], 'Domain must be specified as (lower_bound, upper_bound)'

        return domain

    @field_validator('norm')
    @classmethod
    def _validate_norm(cls, norm: Transformation, info: ValidationInfo) -> list[Transform] | None:
        if norm is None:
            return norm
        norm = Transform.from_string(norm)

        # Set default values for minmax and zscore transforms
        domain = info.data['domain']
        normal_args = None
        if dist := info.data['dist']:
            if isinstance(dist, Normal):
                normal_args = dist.dist_args
        for transform in norm:
            if isinstance(transform, Minmax):
                if domain and np.any(np.isnan(transform.transform_args[0:2])):
                    transform.update(lb=domain[0], ub=domain[1])
            elif isinstance(transform, Zscore):
                if normal_args and np.any(np.isnan(transform.transform_args)):
                    transform.update(mu=normal_args[0], std=normal_args[1])

        return norm

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """Consider two Variables equal if they share the same string id.

        Also returns true when checking if this Variable is equal to a string id by itself.
        """
        if isinstance(other, Variable):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    def update(self, new_attrs: dict = None, **kwargs):
        """Update this variable with `new_attrs` or any specific attributes via `update(domain=...)` for example.

        :param new_attrs: the Variable attributes to update
        """
        if new_attrs:
            for attr, value in new_attrs.items():
                setattr(self, attr, value)
        if kwargs:
            for attr, value in kwargs.items():
                setattr(self, attr, value)

    def get_tex(self, units: bool = False, symbol: bool = True) -> str:
        """Return a raw string that is well-formatted for plotting (with latex).

        :param units: whether to include the units in the string
        :param symbol: just latex symbol if true, otherwise the full description
        """
        s = (self.tex if symbol else self.description) or self.name
        return r'{} [{}]'.format(s, self.units) if units else r'{}'.format(s)

    def get_nominal(self, transform: bool = False) -> float | None:
        """Return the nominal value of the variable. Defaults to the mean for a normal distribution or the
        center of the domain if `var.nominal` is not specified.

        :param transform: return the nominal value in transformed space using `Variable.norm`
        """
        nominal = self.normalize(self.nominal) if transform else self.nominal
        if nominal is None:
            if dist := self.dist:
                dist_args = self.normalize(dist.dist_args) if transform else dist.dist_args
                nominal = dist.nominal(dist_args=dist_args)
            elif domain := self.get_domain(transform=transform):
                nominal = (domain[0] + domain[1]) / 2

        return nominal

    def get_domain(self, transform: bool = False) -> tuple | None:
        """Return a tuple of the defined domain of this variable.

        :param transform: return the domain of the transformed space instead
        """
        if self.domain is None:
            return None
        return tuple(self.normalize(self.domain)) if transform else self.domain

    def sample_domain(self, shape: tuple | int, transform: bool = False) -> np.ndarray:
        """Return an array of the given `shape` for random samples over the domain of this variable.

        :param shape: the shape of samples to return
        :param transform: whether to sample in the transformed space instead
        :returns: the random samples over the domain of the variable
        """
        if isinstance(shape, int):
            shape = (shape, )
        if domain := self.get_domain(transform=transform):
            return np.random.rand(*shape) * (domain[1] - domain[0]) + domain[0]
        else:
            raise RuntimeError(f'Variable "{self.name}" does not have a domain specified.')

    def pdf(self, x: np.ndarray, transform: bool = False) -> np.ndarray:
        """Compute the PDF of the Variable at the given `x` locations.

        !!! Note
            If `transform=True`, then `x` and the arguments of `self.dist` will both be transformed via `self.norm`
            _before_ computing the PDF. Note that this
            means if `x=Variable(dist=N(0, 1), norm=linear(m=2, b=2))`, then samples of $x$ are _not_ normally
            distributed as `N(0,1)`; rather, a new variable $y$ is distributed as $y\\sim\\mathcal{N}(2, 4)$ -- i.e.
            the `dist` parameters get transformed, not the distribution itself. `x.pdf(transform=True)` will then
            actually return the pdf of the transformed variable $y$. This way, for example,
            a log-uniform variable can be obtained via `Variable(dist=U(1, 10), norm=log10)`.

        :param x: locations to compute the PDF at
        :param transform: whether to compute the PDF in transformed space instead
        :returns: the PDF evaluations at `x`
        """
        y = self.normalize(x) if transform else x
        if dist := self.dist:
            dist_args = self.normalize(dist.dist_args) if transform else dist.dist_args
            return dist.pdf(y, dist_args=dist_args)
        else:
            return np.ones(x.shape)  # No pdf if no dist is specified

    def sample(self, shape: tuple | int, nominal: float | np.ndarray = None, transform: bool = False) -> np.ndarray:
        """Draw samples from this Variable's distribution. Just returns the nominal value of the given shape if
        this Variable has no distribution.

        !!! Note
            If `transform=True`, then samples will be drawn in this Variable's transformed space. This would be
            used for example if sampling a log-uniform distributed variable via
            `x=Variable(dist=U(1, 10), norm=log); x.sample(transform=True)`.

        :param shape: the shape of the returned samples
        :param nominal: a nominal value to use if applicable (i.e. a center for relative, tolerance, or normal)
        :param transform: whether to sample in the Variable's transformed space instead
        :returns: samples from the PDF of this Variable's distribution
        """
        if isinstance(shape, int):
            shape = (shape, )
        nominal = (self.normalize(nominal) if transform else nominal) or self.get_nominal(transform=transform)

        if dist := self.dist:
            dist_args = self.normalize(dist.dist_args) if transform else dist.dist_args
            return dist.sample(shape, nominal, dist_args)
        else:
            # Variable's with no distribution
            if nominal is None:
                raise ValueError(f'Cannot sample "{self.name}" with no dist or nominal value specified.')
            else:
                return np.ones(shape) * nominal

    def normalize(self, values: ArrayLike, denorm: bool = False) -> ArrayLike | None:
        """Normalize `values` based on this Variable's `norm` method(s). See `Transform` for available norm methods.

        !!! Note
            If this Variable's `self.norm` was specified as a list of norm methods, then each will be applied in
            sequence in the original order (and in reverse for `denorm=True`). When `self.dist` is involved in the
            transforms (only for minmax and zscore), the `dist_args` will get normalized too at each transform before
            applying the next transform.

        :param values: the values to normalize (array-like)
        :param denorm: whether to denormalize instead using the inverse of the original normalization method
        :returns: the normalized (or unnormalized) values
        """
        if not self.norm or values is None:
            return values
        if dist := self.dist:
            normal_dist = isinstance(dist, Normal)
        else:
            normal_dist = False

        def _normalize_single(values, transform, inverse, domain, dist_args):
            """Do a single transform. Might need to override transform_args depending on the transform."""
            transform_args = None
            if isinstance(transform, Minmax) and domain:
                transform_args = domain + transform.transform_args[2:]  # Update minmax bounds
            elif isinstance(transform, Zscore) and dist_args:
                transform_args = dist_args                              # Update N(mu, std)

            return transform.transform(values, inverse=inverse, transform_args=transform_args)

        domain, dist_args = self.get_domain() or [], self.dist.dist_args if normal_dist else []

        if denorm:
            # First, send domain and dist_args through the forward norm list (up until the last norm)
            hyperparams = [np.hstack((domain, dist_args))]
            for i, transform in enumerate(self.norm):
                domain, dist_args = tuple(hyperparams[i][:2]), tuple(hyperparams[i][2:])
                hyperparams.append(_normalize_single(hyperparams[i], transform, False, domain, dist_args))

            # Now denormalize in reverse
            hp_idx = -2
            for transform in reversed(self.norm):
                domain, dist_args = tuple(hyperparams[hp_idx][:2]), tuple(hyperparams[hp_idx][2:])
                values = _normalize_single(values, transform, True, domain, dist_args)
                hp_idx -= 1
        else:
            # Normalize values and hyperparams through the forward norm list
            hyperparams = np.hstack((domain, dist_args))
            for transform in self.norm:
                domain, dist_args = tuple(hyperparams[:2]), tuple(hyperparams[2:])
                values = _normalize_single(values, transform, denorm, domain, dist_args)
                hyperparams = _normalize_single(hyperparams, transform, denorm, domain, dist_args)
        return values

    def denormalize(self, values):
        """Alias for `normalize(denorm=True)`"""
        return self.normalize(values, denorm=True)

    def compress(self, values: CompressionData, coord: np.ndarray = None,
                 reconstruct: bool = False) -> CompressionData:
        """Compress or reconstruct field quantity values using this Variable's compression info.

        !!! Note "Specifying compression values"
            If only one field quantity is associated with this variable, then `len(field[quantities])=1`. In this case,
            specify `values` as `dict(coord=..., name=...)` for this Variable's `name`. If `coord` is not specified,
            then this assumes the locations are the same as the reconstruction data (and skips interpolation).

        !!! Info "Compression workflow"
            Generally, compression follows `interpolate -> normalize -> compress` to take raw values into the compressed
            "latent" space. The interpolation step is required to make sure `values` align with the coordinates used
            when building the compression map in the first place (such as through SVD).

        :param values: a `dict` with a key for each field qty of shape `(..., qty.shape)` and a `coord` key of shape
                      `(qty.shape, dim)` that gives the coordinates of each point. Only a single `latent` key should
                      be given instead if `reconstruct=True`.
        :param coord: the coordinates of each point in `values` if `values` did not contain a `coord` key;
                       defaults to the compression grid coordinates
        :param reconstruct: whether to reconstruct values instead of compress
        :returns: the compressed values with key `latent` and shape `(..., latent_size)`; if `reconstruct=True`,
                  then the reconstructed values with shape `(..., qty.shape)` for each `qty` key are returned.
                  The return `dict` also has a `coord` key with shape `(qty.shape, dim)`.
        """
        if not self.compression:
            raise ValueError(f'Compression is not supported for the non-field variable "{self.name}".')
        if not self.compression.map_exists:
            raise ValueError(f'Compression map not computed yet for "{self.name}".')

        # Default field coordinates to the compression coordinates if they are not provided
        field_coords = values.pop('coord', coord)
        if field_coords is None:
            field_coords = self.compression.coords
        ret_dict = {'coord': field_coords}

        # For reconstruction: decompress -> denormalize -> interpolate
        if reconstruct:
            try:
                states = np.atleast_1d(values['latent'])    # (..., rank)
            except KeyError as e:
                raise ValueError('Must pass values["latent"] in for reconstruction.') from e
            states = self.compression.reconstruct(states)   # (..., dof)
            states = self.denormalize(states)               # (..., dof)
            states = self.compression.interpolate_from_grid(states, field_coords)
            ret_dict.update(states)

        # For compression: interpolate -> normalize -> compress
        else:
            states = self.compression.interpolate_to_grid(field_coords, values)
            states = self.normalize(states)                 # (..., dof)
            states = self.compression.compress(states)      # (..., rank)
            ret_dict['latent'] = states

        return ret_dict

    def reconstruct(self, values, coord=None):
        """Alias for `compress(reconstruct=True)`"""
        return self.compress(values, coord=coord, reconstruct=True)

    @property
    def shape(self):
        return self.compression.shape if self.compression else ()

    def serialize(self, save_path: str | Path = '.') -> dict:
        """Convert a `Variable` to a `dict` with only standard Python types
        (i.e. convert custom objects like `dist` and `norm` to strings).
        """
        instance_variables = {key: value for key, value in self.__dict__.items() if value is not None}
        if domain := instance_variables.get('domain'):
            instance_variables['domain'] = str(domain)
        if dist := instance_variables.get('dist'):
            instance_variables['dist'] = str(dist)
        if norm := instance_variables.get('norm'):
            instance_variables['norm'] = str(norm) if isinstance(norm, str | Transform) else [str(transform) for
                                                                                              transform in norm]
        if compression := instance_variables.get('compression'):
            fname = f'{self.name}_compression.pkl'
            instance_variables['compression'] = compression.serialize(save_path=Path(save_path) / fname)
        return instance_variables

    @classmethod
    def deserialize(cls, data: dict, search_paths=None) -> Variable:
        """Convert a `dict` to a `Variable` object. Let `pydantic` handle validation and conversion of fields."""
        if isinstance(data, Variable):
            return data
        else:
            if (compression := data.get('compression', None)) is not None:
                if isinstance(compression, str):
                    data['compression'] = search_for_file(compression, search_paths=search_paths)
            return cls(**data)

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, data: Variable) -> yaml.MappingNode:
        """Convert a single `Variable` object (`data`) to a yaml MappingNode (i.e. a `dict`)."""
        save_path, save_file = _get_yaml_path(dumper)
        return dumper.represent_mapping(Variable.yaml_tag, data.serialize(save_path=save_path))

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the `!Variable` tag in yaml to a single `Variable` object (or a list of `Variables`)."""
        save_path, save_file = _get_yaml_path(loader)
        if isinstance(node, yaml.SequenceNode):
            return [ele if isinstance(ele, Variable) else Variable.deserialize(ele, search_paths=[save_path]) for ele in
                    loader.construct_sequence(node, deep=True)]
        elif isinstance(node, yaml.MappingNode):
            return Variable.deserialize(loader.construct_mapping(node), search_paths=[save_path])
        else:
            raise NotImplementedError(f'The "{Variable.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')


class VariableList(OrderedDict, Serializable):
    """Store Variables as `str(var) : Variable` in the order they were passed in. You can:

    - Initialize/update from a single Variable or a list of Variables
    - Get/set a Variable directly or by name via `my_vars[var]` or `my_vars[str(var)]` etc.
    - Retrieve the original order of insertion by `list(my_vars.items())`
    - Access/delete elements by order of insertion using integer/slice indexing (i.e. `my_vars[1:3]`)
    - Save/load from yaml file using the `!VariableList` tag
     """
    yaml_tag = '!VariableList'

    def __init__(self, data: list[Variable] | Variable | OrderedDict | dict = None, **kwargs):
        """Initialize a collection of `Variable` objects."""
        super().__init__()
        self.update(data, **kwargs)

    def __iter__(self):
        yield from self.values()

    def __eq__(self, other):
        if isinstance(other, VariableList):
            for v1, v2 in zip(self.values(), other.values()):
                if v1 != v2:
                    return False
            return True
        else:
            return False

    def append(self, data: Variable):
        self.update(data)

    def extend(self, data: list[Variable]):
        self.update(data)

    def index(self, key):
        for i, k in enumerate(self.keys()):
            if k == key:
                return i
        raise ValueError(f"'{key}' is not in list")

    def update(self, data: list[Variable] | Variable | OrderedDict | dict = None, **kwargs):
        """Update from a list or dict of `Variable` objects, or from `key=value` pairs."""
        if data:
            if isinstance(data, OrderedDict | dict):
                for key, value in data.items():
                    self.__setitem__(key, value)
            else:
                data = [data] if not isinstance(data, list) else data
                for variable in data:
                    self.__setitem__(str(variable), variable)
        if kwargs:
            for key, value in kwargs.items():
                self.__setitem__(key, value)

    def get(self, key, default=None):
        """Make sure this passes through `__getitem__()`"""
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def __setitem__(self, key, value):
        """Only allow `str(var): Variable` items. Or normal list indexing via `my_vars[0] = var`."""
        if isinstance(key, int):
            k = list(self.keys())[key]
            self.__setitem__(k, value)
            return
        if not isinstance(key, str | Variable):
            raise TypeError(f'VariableList key "{key}" is not a Variable or string.')
        if not isinstance(value, Variable):
            raise TypeError(f'VariableList value "{value}" is not a Variable.')
        super().__setitem__(str(key), value)

    def __getitem__(self, key):
        """Allow accessing variable(s) directly via `my_vars[var]` or by index/slicing."""
        if isinstance(key, list | tuple):
            return [self.__getitem__(ele) for ele in key]
        elif isinstance(key, int | slice):
            return list(self.values())[key]
        elif isinstance(key, str | Variable):
            return super().__getitem__(str(key))
        else:
            raise TypeError(f'VariableList key "{key}" is not valid.')

    def __delitem__(self, key):
        """Allow deleting variable(s) directly or by index/slicing."""
        if isinstance(key, list | tuple):
            for ele in key:
                self.__delitem__(ele)
        elif isinstance(key, int | slice):
            ele = list(self.keys())[key]
            if isinstance(ele, list):
                for item in ele:
                    super().__delitem__(item)
            else:
                super().__delitem__(ele)
        elif isinstance(key, str | Variable):
            super().__delitem__(str(key))
        else:
            raise TypeError(f'VariableList key "{key}" is not valid.')

    def __str__(self):
        return str(list(self.values()))

    def __repr__(self):
        return self.__str__()

    def serialize(self, save_path='.') -> list[dict]:
        return [var.serialize(save_path=save_path) for var in self.values()]

    @classmethod
    def deserialize(cls, data: dict | list[dict], search_paths=None) -> VariableList:
        if not isinstance(data, list):
            data = [data]
        return cls([Variable.deserialize(d, search_paths=search_paths) for d in data])

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, data: VariableList) -> yaml.SequenceNode:
        """Convert a single `VariableList` object (`data`) to a yaml SequenceNode (i.e. a list)."""
        save_path, save_file = _get_yaml_path(dumper)
        return dumper.represent_sequence(VariableList.yaml_tag, data.serialize(save_path=save_path))

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the `!VariableList` tag in yaml to a `VariableList` object."""
        save_path, save_file = _get_yaml_path(loader)
        if isinstance(node, yaml.SequenceNode):
            return VariableList.deserialize(loader.construct_sequence(node, deep=True), search_paths=[save_path])
        elif isinstance(node, yaml.MappingNode):
            return VariableList.deserialize(loader.construct_mapping(node), search_paths=[save_path])
        else:
            raise NotImplementedError(f'The "{VariableList.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')
