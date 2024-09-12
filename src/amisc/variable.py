"""Provides an object-oriented interface for model inputs/outputs and random variables.

Includes:

- `Distribution` — an object for specifying a PDF. `Normal`, `Uniform`, `Relative`, and `Tolerance` are available.
- `Transform` — an object for specifying a transformation. `Linear`, `Log`, `Minmax`, and `Zscore` are available.
- `FieldQuantity` — a dictionary spec for variables that contain field quantity data (i.e. high-dimensional outputs)
- `CompressionData` — a dictionary spec for passing data to/from `Variable.compress()`
- `Variable` — an object that stores information about a variable and includes methods for sampling, pdf evaluation,
               normalization, compression, loading from file, etc.
- `VariableList` — a container for Variables that provides dict-like access of Variables by `var_id` along with normal
                   indexing and slicing

The preferred serialization of `Variable` and `VariableList` is to/from yaml. `Distribution` and `Transform`
objects can be serialized via conversion to/from string.
"""
from __future__ import annotations

import ast
import random
import string
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

import h5py
import numpy as np
import yaml
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, ValidationInfo, field_validator
from scipy.interpolate import RBFInterpolator
from typing_extensions import TypedDict


def _parse_function_string(func_string: str) -> tuple[str, list, dict]:
    """Convert a function signature like `func(a, b, key=value)` to name, args, kwargs."""
    try:
        parts = func_string.strip().split('(')
        func_name = parts[0]
        all_args = parts[1].split(')')[0].split(',') if len(parts) > 1 else []
        args, kwargs = [], {}
        for arg in all_args:
            if arg in ['*', '/']:
                continue
            if '=' in arg:
                key, value = arg.split('=')
                kwargs[key.strip()] = value.strip()
            else:
                args.append(arg.strip())
        return func_name, args, kwargs
    except Exception as e:
        raise ValueError(f'Function string "{func_string}" is not valid.') from e


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
                            `Relative(pct)`, or `Tolerance(tol)`. The shorthands `N(0, 1)`, `U(0, 1)`, `rel(5%)`, or
                            `tol(1)` are also accepted.
        :return: the corresponding `Distribution` object
        """
        if not dist_string:
            return None

        dist_name, args, kwargs = _parse_function_string(dist_string)
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
                pct = float(kwargs.get('pct', args[0].split(r'%')[0].strip()))
                return Relative((pct,))
            except Exception as e:
                raise ValueError(f'Relative distribution string "{dist_string}" is not valid: Try rel(5%).') from e
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
        return rf'rel({self.dist_args[0]}%)'

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


class FieldQuantity(TypedDict, total=False):
    """Configuration `dict` for field quantities.

    TODO: Make this interface more general and check that all compression data is provided and in the right format

    :ivar quantities: a list of quantities included with this FieldQuantity (e.g. `v_x, v_y, v_z` for 3d velocity)
    :ivar compress_datapath: the path to the compression data
    :ivar compress_method: the compression method
    :ivar compress_kwargs: extra args for `Variable.compress()`
    :ivar interp_kwargs: extra args for the `RBFInterpolator` used during compression
    """
    quantities: list[str]
    compress_datapath: Union[FilePath, DirectoryPath, None]
    compress_method: Literal['svd']
    compress_kwargs: dict[str, Any]
    interp_kwargs: dict[str, Any]


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
    :ivar latent: `(..., latent_size)` array of latent space coefficients for a `FieldQuantity`; this is what is
                  _returned_ by `Variable.compress()` and what is _expected_ as input by `Variable.reconstruct()`.
    :ivar qty: `(..., qty.shape)` array of uncompressed field quantity data for this qty within
               the `FieldQuantity[quantities]` list. Each qty in this list will be its own `key:value` pair in the
               `CompressionData` structure
    """
    coord: np.ndarray
    latent: np.ndarray
    qty: np.ndarray


class Transform(ABC):
    """A base class for all transformations.

    :ivar transform_args
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

        - **linear** — $x_{norm} = mx + b$ specified as `linear(m, b)` or `linear(slope=m, offset=b)`. `b=0` if not
                       specified. Use `m=1/10` for fractional values (for example).
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

            name, args, kwargs = _parse_function_string(spec_string)
            if name.lower() == 'linear':
                try:
                    slope = kwargs.get('slope', args[0] if len(args) > 0 else '1')
                    slope = float(slope.split('/')[0]) / float(slope.split('/')[1]) if '/' in slope else float(slope)
                    offset = float(kwargs.get('offset', args[1] if len(args) > 1 else 0))
                    transforms.append(Linear((slope, offset)))
                except Exception as e:
                    raise ValueError(f'Linear transform spec "{spec_string}" is not valid: Try "linear(m, b)".') from e
            elif name.lower() in ['log', 'log10']:
                try:
                    log_base = kwargs.get('base', args[0] if len(args) > 0 else ('e' if name.lower() == 'log' else 10))
                    log_base = np.e if log_base == 'e' else float(log_base)
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


class Variable(BaseModel):
    """Object for storing information about variables and providing methods for pdf evaluation, sampling, etc.

    A simple variable object can be created with `var = Variable()`. All initialization options are optional and will
    be given good defaults. You should probably at the very least give a memorable `var_id` and a `domain`.

    With the `pyyaml` library installed, all Variable objects can be saved or loaded directly from a `.yml` file by
    using the `!Variable` yaml tag (which is loaded by default with `amisc`).

    - Use `Variable.dist` to specify sampling PDFs, such as for random variables. See the `Distribution` classes.
    - Use `Variable.norm` to specify a transformed-space that is more amenable to surrogate construction
      (e.g. mapping to the range (0,1)). See the `Transform` classes.
    - Use `Variable.field` to specify high-dimensional, coordinate-based field quantities, such as from the output of
      many simulation software programs. See `FieldQuantity` for arguments.
    - Use `Variable.category` as an additional layer for using Variable's in different ways (e.g. set a "calibration"
      category for Bayesian inference).

    !!! Example
        ```python
        # Random variable
        temp = Variable(var_id='T', description='Temperature', units='K', dist='Uniform(280, 320)')
        samples = temp.sample(100)
        pdf = temp.pdf(samples)

        # Field quantity
        vel = Variable(var_id='u', description='Velocity', units='m/s', field={'quantities': ['ux', 'uy', 'uz']})
        vel_data = ...  # from a simulation
        reduced_vel = vel.compress(vel_data)
        ```

    !!! Warning
        Changes to collection fields (like `Variable.field` and `Variable.norm`) should completely reassign the _whole_
        collection to trigger the correct validation, rather than editing particular entries. For example, reassign
        `norm=['log', 'linear(2, 2)']` rather than editing norm via `norm.append('linear(2, 2)')`.

    :ivar var_id: an identifier for the variable, can compare variables directly with strings for indexing purposes
    :ivar nominal: a typical value for this variable
    :ivar description: a lengthier description of the variable
    :ivar units: assumed units for the variable (if applicable)
    :ivar category: an additional descriptor for how this variable is used, e.g. calibration, operating, design, etc.
    :ivar tex: latex format for the variable, i.e. r"$x_i$"
    :ivar field: specifies field quantities and links to relevant compression data
    :ivar dist: a string specifier of a probability distribution function (or a `Distribution` object)
    :ivar domain: the explicit domain bounds of the variable (limits of where you expect to use it)
    :ivar norm: specifier of a map to a transformed-space for surrogate construction (or a `Transformation` type)

    :vartype var_id: str
    :vartype nominal: float
    :vartype description: str
    :vartype units: str
    :vartype category: str
    :vartype tex: str
    :vartype field: FieldQuantity
    :vartype dist: str | Distribution
    :vartype domain: tuple[float, float]
    :vartype norm: Transformation
    """
    yaml_tag: ClassVar[str] = u'!Variable'
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, validate_default=True)

    var_id: Annotated[str, Field(default_factory=lambda: "X_" + "".join(random.choices(string.digits, k=3)))]
    nominal: Optional[float] = None
    description: Optional[str] = None
    units: Optional[str] = None
    category: Optional[str] = None
    tex: Optional[str] = None
    field: Optional[FieldQuantity] = None
    dist: Optional[str | Distribution] = None
    domain: Optional[str | tuple[float, float]] = None
    norm: Optional[Transformation] = None

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

    @field_validator('field')
    @classmethod
    def _validate_field(cls, field: dict, info: ValidationInfo) -> dict | None:
        if field is None:
            return field
        else:
            field['quantities'] = field.get('quantities', [info.data['var_id']])
            field['compress_datapath'] = field.get('compress_datapath', None)
            field['compress_method'] = field.get('compress_method', 'svd')
            field['compress_kwargs'] = field.get('compress_kwargs', dict())
            field['interp_kwargs'] = field.get('interp_kwargs', dict())

        if datapath := field.get('compress_datapath'):
            assert Path(datapath).exists()
        return field

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
        return self.var_id

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.var_id)

    def __eq__(self, other):
        """Consider two Variables equal if they share the same string id.

        Also returns true when checking if this Variable is equal to a string id by itself.
        """
        if isinstance(other, Variable):
            return self.var_id == other.var_id
        elif isinstance(other, str):
            return self.var_id == other
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
        s = (self.tex if symbol else self.description) or self.var_id
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
            raise RuntimeError(f'Variable "{self.var_id}" does not have a domain specified.')

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
                raise ValueError(f'Cannot sample "{self.var_id}" with no dist or nominal value specified.')
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

    def compress(self, values: CompressionData, reconstruct: bool = False,
                 interp_kwargs: dict = None) -> CompressionData:
        """Compress or reconstruct field quantity values using this Variable's compression info.

        !!! Note "Specifying compression values"
            If only one field quantity is associated with this variable, then `len(field[quantities])=0`. In this case,
            specify `values` as `dict(coord=..., var_id=...)` for this Variable's `var_id`. If `coord` is not specified,
            then this assumes the locations are the same as the reconstruction data (and skips interpolation).

        !!! Info "Compression workflow"
            Generally, compression follows `interpolate -> normalize -> compress` to take raw values into the compressed
            "latent" space. The interpolation step is required to make sure `values` align with the coordinates used
            when building the compression map in the first place (such as through SVD). Use compression functions
            provided in `amisc.compress` to ensure proper formatting of the compression data.

        :param values: a `dict` with a key for each field qty of shape `(..., qty.shape)` and a `coord` key of shape
                      `(qty.shape, dim)` that gives the coordinates of each point. Only a single `latent` key should
                      be given instead if `reconstruct=True`.
        :param reconstruct: whether to reconstruct values instead of compress
        :param interp_kwargs: extra arguments passed to `scipy.interpolate.RBFInterpolator` for interpolating between
                              the reconstruction grid coordinates and the passed-in coordinates
        :returns: the compressed values with key `latent` and shape `(..., latent_size)`; if `reconstruct=True`,
                  then the reconstructed values with shape `(..., qty.shape)` for each `qty` key are returned.
                  The return `dict` also has a `coord` key with shape `(qty.shape, dim)`.
        """
        field = self.field
        if not field:
            raise ValueError(f'Compression is not supported for the non-field variable "{self.var_id}".')
        if not field.get('compress_datapath'):
            raise ValueError(f'Compression datapath not specified for variable "{self.var_id}".')
        method = field.get('compress_method', 'svd')
        datapath = Path(field['compress_datapath'])
        interp_kwargs = interp_kwargs or field.get('interp_kwargs', dict())
        if not datapath.exists():
            raise ValueError(f'The compression datapath "{datapath}" for variable "{self.var_id}" does not exist.')

        qty_list = field.get('quantities', [self.var_id])
        nqoi = len(qty_list)

        if method.lower() == 'svd':
            if datapath.suffix not in ['.h5', '.hdf5']:
                raise ValueError(f'Compression file type "{datapath.suffix}" is not supported for SVD. '
                                 f'Only [.h5, .hdf5].')
            with h5py.File(datapath, 'r') as fd:
                try:
                    projection_matrix = fd['projection_matrix'][:]   # (dof, rank)
                    rec_coords = fd['coord'][:]                      # (num_pts, dim)
                    num_pts, dim = rec_coords.shape                  # Also num_pts = dof / nqoi
                    reconstruct_func = lambda latent: np.squeeze(projection_matrix @ latent[..., np.newaxis], axis=-1)
                    compress_func = lambda states: np.squeeze(projection_matrix.T @ states[..., np.newaxis], axis=-1)
                except Exception as e:
                    raise ValueError('SVD compression data not formatted correctly.') from e
        else:
            raise NotImplementedError(f'The compression method "{method}" is not implemented.')

        # Default field coordinates to the reconstruction coordinates if they are not provided
        if values.get('coord') is not None:
            no_interp = False
            field_coords = np.atleast_1d(values.get('coord'))
            if len(field_coords.shape) == 1:
                field_coords = field_coords[..., np.newaxis] if dim == 1 else field_coords[np.newaxis, ...]
        else:
            no_interp = True
            field_coords = rec_coords
        qty_shape = field_coords.shape[:-1]
        field_coords = field_coords.reshape(-1, dim)    # (N, dim)

        # Don't need to interpolate if passed-in coords are same as reconstruction coords
        no_interp = no_interp or (field_coords.shape == rec_coords.shape and np.allclose(field_coords, rec_coords))
        ret_dict = {'coord': values.get('coord', rec_coords)}

        # For reconstruction: decompress -> denormalize -> interpolate
        if reconstruct:
            try:
                latent = np.atleast_1d(values['latent'])                                # (..., rank)
            except KeyError as e:
                raise ValueError('Must pass values["latent"] in for reconstruction.') from e
            states = reconstruct_func(latent)                                           # (..., dof)
            states = self.denormalize(states).reshape((*states.shape[:-1], num_pts, nqoi))

            for i, qoi in enumerate(qty_list):
                if no_interp:
                    ret_dict[qoi] = states[..., i]
                else:
                    reshaped_states = states[..., i].reshape(-1, num_pts).T                     # (num_pts, N...)
                    rbf = RBFInterpolator(rec_coords, reshaped_states, **interp_kwargs)
                    field_vals = rbf(field_coords).T.reshape(*states.shape[:-2], *qty_shape)    # (..., qty.shape)
                    ret_dict[qoi] = field_vals

        # For compression: interpolate -> normalize -> compress
        else:
            states = None
            for i, qoi in enumerate(qty_list):
                if states is None:
                    states = np.empty((*values[qoi].shape[:-len(qty_shape)], num_pts, nqoi))
                field_vals = values[qoi].reshape((*states.shape[:-2], -1))              # (..., N)
                if no_interp:
                    states[..., i] = field_vals
                else:
                    reshaped_field = field_vals.reshape(-1, field_vals.shape[-1]).T     # (N, ...)
                    rbf = RBFInterpolator(field_coords, reshaped_field, **interp_kwargs)
                    states[..., i] = rbf(rec_coords).T.reshape(*states.shape[:-2], num_pts)
            states = self.normalize(np.reshape(states, (*states.shape[:-2], -1)))       # (..., dof)
            latent = compress_func(states)                                              # (..., rank)
            ret_dict['latent'] = latent

        return ret_dict

    def reconstruct(self, values, **kwargs):
        """Alias for `compress(reconstruct=True)`"""
        return self.compress(values, reconstruct=True, **kwargs)

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, data) -> yaml.MappingNode:
        """Convert a single `Variable` object (`data`) to a yaml MappingNode (i.e. a dict)."""
        instance_variables = {key: value for key, value in data.__dict__.items() if value is not None}
        if domain := instance_variables.get('domain'):
            instance_variables['domain'] = str(domain)
        if dist := instance_variables.get('dist'):
            instance_variables['dist'] = str(dist)
        if norm := instance_variables.get('norm'):
            instance_variables['norm'] = str(norm) if isinstance(norm, str | Transform) else [str(transform) for
                                                                                              transform in norm]
        return dumper.represent_mapping(Variable.yaml_tag, instance_variables)

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the !Variable tag in yaml to a single `Variable` object (or a list of `Variables`)."""
        if isinstance(node, yaml.SequenceNode):
            return [ele if isinstance(ele, Variable) else Variable(**ele) for ele in
                    loader.construct_sequence(node, deep=True)]
        elif isinstance(node, yaml.MappingNode):
            return Variable(**loader.construct_mapping(node))
        else:
            raise NotImplementedError(f'The "{Variable.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')


class VariableList(OrderedDict):
    """Store Variables as `str(var) : Variable` in the order they were passed in. You can:

    - Initialize/update from a single Variable or a list of Variables
    - Get/set a Variable directly or by var_id via `my_vars[var]` or `my_vars[str(var)]` etc.
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

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, data) -> yaml.SequenceNode:
        """Convert a single `VariableList` object (`data`) to a yaml SequenceNode (i.e. a list)."""
        return dumper.represent_sequence(VariableList.yaml_tag, list(data.values()))

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the !VariableList tag in yaml to a `VariableList` object."""
        if isinstance(node, yaml.SequenceNode):
            variables = [ele if isinstance(ele, Variable) else Variable(**ele) for ele in
                         loader.construct_sequence(node, deep=True)]
            return VariableList(variables)
        elif isinstance(node, yaml.MappingNode):
            return VariableList(Variable(**loader.construct_mapping(node)))
        else:
            raise NotImplementedError(f'The "{Variable.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')
