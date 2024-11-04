"""Provides an object-oriented interface for model inputs/outputs, random variables, scalars, and field quantities.

Includes:

- `Variable` — an object that stores information about a variable and includes methods for sampling, pdf evaluation,
               normalization, compression, loading from file, etc. Variables can mostly be treated as strings
               that have some additional information and utilities attached to them.
- `VariableList` — a container for `Variables` that provides dict-like access of `Variables` by `name` along with normal
                   indexing and slicing.

The preferred serialization of `Variable` and `VariableList` is to/from yaml. This is done by default with the
`!Variable` and `!VariableList` yaml tags.
"""
from __future__ import annotations

import ast
import random
import string
from collections import OrderedDict
from pathlib import Path
from typing import ClassVar, Optional, Union

import numpy as np
import yaml
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from amisc.compression import Compression
from amisc.distribution import Distribution, LogUniform, Normal, Uniform
from amisc.serialize import Serializable
from amisc.transform import Minmax, Transform, Zscore
from amisc.typing import LATENT_STR_ID, CompressionData
from amisc.utils import _get_yaml_path, _inspect_assignment, search_for_file

__all__ = ['Variable', 'VariableList']
_TransformLike = Union[str, Transform, list[str | Transform]]  # something that can be converted to a Transform


class Variable(BaseModel, Serializable):
    """Object for storing information about variables and providing methods for pdf evaluation, sampling, etc.
    All fields will undergo pydantic validation and conversion to the correct types.

    A simple variable object can be created with `var = Variable()`. All initialization options are optional and will
    be given good defaults. You should probably at the very least give a memorable `name` and a `domain`. Variables
    can mostly be treated as strings with some extra information/utilities attached.

    With the `pyyaml` library installed, all `Variable` objects can be saved or loaded directly from a `.yml` file by
    using the `!Variable` yaml tag (which is loaded by default with `amisc`).

    - Use `Variable.distribution` to specify PDFs, such as for random variables. See the `Distribution` classes.
    - Use `Variable.norm` to specify a transformed-space that is more amenable to surrogate construction
      (e.g. mapping to the range (0,1)). See the `Transform` classes.
    - Use `Variable.compression` to specify high-dimensional, coordinate-based field quantities,
      such as from the output of many simulation software programs. See the `Compression` classes.
    - Use `Variable.category` as an additional layer for using Variable's in different ways (e.g. set a "calibration"
      category for Bayesian inference).

    !!! Example
        ```python
        # Random variable
        temp = Variable(name='T', description='Temperature', units='K', distribution='Uniform(280, 320)')
        samples = temp.sample(100)
        pdf = temp.pdf(samples)

        # Field quantity
        vel = Variable(name='u', description='Velocity', units='m/s', compression={'fields': ['ux', 'uy', 'uz']})
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
    :ivar tex: latex format for the variable, i.e. "$x_i$"
    :ivar compression: specifies field quantities and links to relevant compression data
    :ivar distribution: a string specifier of a probability distribution function (see the `Distribution` types)
    :ivar domain: the explicit domain bounds of the variable (limits of where you expect to use it);
                  for field quantities, this is a list of domains for each latent dimension
    :ivar norm: specifier of a map to a transformed-space for surrogate construction (see the `Transform` types)
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
    distribution: Optional[str | Distribution] = None
    domain: Optional[str | tuple[float, float] | list] = None
    norm: Optional[_TransformLike] = None

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

    @field_validator('distribution')
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
    def _validate_domain(cls, domain: list | tuple | str, info: ValidationInfo) -> tuple | list | None:
        """Try to extract the domain from the distribution if not provided, or convert from a string.
        Returns a list of domains for each latent dimension if this is a field quantity with compression.
        """
        if domain is None:
            if dist := info.data['distribution']:
                domain = dist.domain()
            elif compression := info.data['compression']:
                domain = compression.estimate_latent_ranges()
        elif isinstance(domain, str):
            domain = tuple(ast.literal_eval(domain.strip()))
        elif isinstance(domain, list):
            domain = [tuple(ast.literal_eval(d.strip())) if isinstance(d, str) else d for d in domain]

        if domain is None:
            return domain

        if isinstance(domain, list):
            for d in domain:
                assert isinstance(d, tuple) and len(d) == 2
                assert d[1] > d[0], 'Domain must be specified as (lower_bound, upper_bound)'
        else:
            assert isinstance(domain, tuple) and len(domain) == 2
            assert domain[1] > domain[0], 'Domain must be specified as (lower_bound, upper_bound)'

        return domain

    @field_validator('norm')
    @classmethod
    def _validate_norm(cls, norm: _TransformLike, info: ValidationInfo) -> list[Transform] | None:
        if norm is None:
            return norm
        norm = Transform.from_string(norm)

        # Set default values for minmax and zscore transforms
        domain = info.data['domain']
        normal_args = None
        if dist := info.data['distribution']:
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
        """Allows variables to be used as keys in dictionaries and to be considered equal to their string
        representations.
        """
        return hash(self.name)

    def __eq__(self, other):
        """Consider two `Variables` equal if they share the same string name

        Also returns true when checking if this `Variable` is equal to a string by itself.
        """
        if isinstance(other, Variable):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    def get_tex(self, units: bool = False, symbol: bool = True) -> str:
        """Return a raw string that is well-formatted for plotting (with latex).

        :param units: whether to include the units in the string
        :param symbol: just latex symbol if true, otherwise the full description
        :returns: the latex formatted string
        """
        s = (self.tex if symbol else self.description) or self.name
        return r'{} [{}]'.format(s, self.units or '-') if units else r'{}'.format(s)

    def get_nominal(self) -> float | list | None:
        """Return the nominal value of the variable. Defaults to the mean for a normal distribution or the
        center of the domain if `var.nominal` is not specified. Returns a list of nominal values for each latent
        dimension if this is a field quantity with compression.

        :returns: the nominal value(s)
        """
        nominal = self.nominal
        if nominal is None:
            if dist := self.distribution:
                nominal = float(dist.nominal())
            elif domain := self.get_domain():
                nominal = [np.mean(d) for d in domain] if isinstance(domain, list) else float(np.mean(domain))

        return nominal

    def get_domain(self) -> tuple | list | None:
        """Return a tuple of the defined domain of this variable. Returns a list of domains for each latent dimension
        if this is a field quantity with compression.

        :returns: the domain(s) of this variable
        """
        if self.domain is None:
            return None
        elif isinstance(self.domain, list):
            return self.domain
        elif self.compression is not None:
            # Try to infer a list of domains from compression latent size
            try:
                return [self.domain] * self.compression.latent_size()
            except Exception as e:
                raise ValueError(f'Variables with `compression` data should return a list of domains, one '
                                 f'for each latent coefficient. Could not infer domain for "{self.name}".') from e
        else:
            return self.domain

    def sample_domain(self, shape: tuple | int) -> np.ndarray:
        """Return an array of the given `shape` for uniform samples over the domain of this variable. Returns
        samples for each latent dimension if this is a field quantity with compression.

        !!! Note
            The last dim of the returned samples will be the latent space size for field quantities.

        :param shape: the shape of samples to return
        :returns: the random samples over the domain of the variable
        """
        if isinstance(shape, int):
            shape = (shape, )
        if domain := self.get_domain():
            if isinstance(domain, list):
                lb = np.atleast_1d([d[0] for d in domain])
                ub = np.atleast_1d([d[1] for d in domain])
                return np.random.rand(*shape, 1) * (ub - lb) + lb
            else:
                return np.random.rand(*shape) * (domain[1] - domain[0]) + domain[0]
        else:
            raise RuntimeError(f'Variable "{self.name}" does not have a domain specified.')

    def update_domain(self, domain: tuple[float, float] | list[tuple], override: bool = False):
        """Update the domain of this variable by taking the minimum or maximum of the new domain with the current domain
        for the lower and upper bounds, respectively. Will attempt to update the domain of each latent dimension
        if this is a field quantity with compression. If the variable has a `Uniform` distribution, this will
        update the distribution's bounds too.

        :param domain: the new domain(s) to update with
        :param override: will simply set the domain to the new values rather than update against the current domain;
                         (default `False`)
        """
        def _update_domain(domain, curr_domain):
            lb, ub = domain
            ret = (lb, ub) if override else (min(lb, curr_domain[0]) if curr_domain is not None else lb,
                                             max(ub, curr_domain[1]) if curr_domain is not None else ub)
            return tuple(map(float, ret))

        curr_domain = self.get_domain()
        if isinstance(domain, list):
            if not isinstance(curr_domain, list):
                curr_domain = [curr_domain] * len(domain)
            self.domain = [_update_domain(d, curr_domain[i]) for i, d in enumerate(domain)]
        elif isinstance(curr_domain, list):
            if not isinstance(domain, list):
                domain = [domain] * len(curr_domain)
            self.domain = [_update_domain(d, curr_domain[i]) for i, d in enumerate(domain)]
        else:
            self.domain = _update_domain(domain, curr_domain)
            if (dist := self.distribution) is not None and isinstance(dist, Uniform | LogUniform):
                dist.dist_args = self.domain  # keep Uniform dist in sync

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the PDF of the Variable at the given `x` locations. Will just return one's if the variable
        does not have a distribution.

        :param x: locations to compute the PDF at
        :returns: the PDF evaluations at `x`
        """
        if dist := self.distribution:
            return dist.pdf(x)
        else:
            return np.ones(x.shape)  # No pdf if no dist is specified

    def sample(self, shape: tuple | int, nominal: float | np.ndarray = None) -> np.ndarray:
        """Draw samples from this `Variable's` distribution. Just returns the nominal value of the given shape if
        this `Variable` has no distribution.

        :param shape: the shape of the returned samples
        :param nominal: a nominal value to use if applicable (i.e. a center for relative, tolerance, or normal)
        :returns: samples from the PDF of this `Variable's` distribution
        """
        if isinstance(shape, int):
            shape = (shape, )
        if nominal is None:
            nominal = self.get_nominal()

        if dist := self.distribution:
            return dist.sample(shape, nominal)
        else:
            # Variable's with no distribution
            if nominal is None:
                raise ValueError(f'Cannot sample "{self.name}" with no dist or nominal value specified.')
            elif isinstance(nominal, list | np.ndarray):
                return np.ones(shape + (len(nominal),)) * np.atleast_1d(nominal)  # For field quantities
            else:
                return np.ones(shape) * nominal

    def normalize(self, values: ArrayLike, denorm: bool = False) -> ArrayLike | None:
        """Normalize `values` based on this `Variable's` `norm` method(s). See `Transform` for available norm methods.

        !!! Note
            If this Variable's `self.norm` was specified as a list of norm methods, then each will be applied in
            sequence in the original order (and in reverse for `denorm=True`). When `self.distribution` is involved in
            the transforms (only for `minmax` and `zscore`), the `dist_args` will get normalized too at each
            transform before applying the next transform.

        :param values: the values to normalize (array-like)
        :param denorm: whether to denormalize instead using the inverse of the original normalization method
        :returns: the normalized (or unnormalized) values
        """
        if not self.norm or values is None:
            return values
        if dist := self.distribution:
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

        domain = self.get_domain() or ()
        dist_args = self.distribution.dist_args if normal_dist else []
        if isinstance(domain, list):
            domain = ()  # For field quantities, domain is not used in normalization

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
        """Alias for `normalize(denorm=True)`. See `normalize` for more details."""
        return self.normalize(values, denorm=True)

    def compress(self, values: CompressionData, coords: np.ndarray = None,
                 reconstruct: bool = False) -> CompressionData:
        """Compress or reconstruct field quantity values using this `Variable's` compression info.

        !!! Note "Specifying compression values"
            If only one field quantity is associated with this variable, then
            specify `values` as `dict(coords=..., name=...)` for this Variable's `name`. If `coords` is not specified,
            then this assumes the locations are the same as the reconstruction data (and skips interpolation).

        !!! Info "Compression workflow"
            Generally, compression follows `interpolate -> normalize -> compress` to take raw values into the compressed
            "latent" space. The interpolation step is required to make sure `values` align with the coordinates used
            when building the compression map in the first place (such as through SVD).

        :param values: a `dict` with a key for each field qty of shape `(..., qty.shape)` and a `coords` key of shape
                      `(qty.shape, dim)` that gives the coordinates of each point. Only a single `latent` key should
                      be given instead if `reconstruct=True`.
        :param coords: the coordinates of each point in `values` if `values` did not contain a `coords` key;
                       defaults to the compression grid coordinates
        :param reconstruct: whether to reconstruct values instead of compress
        :returns: the compressed values with key `latent` and shape `(..., latent_size)`; if `reconstruct=True`,
                  then the reconstructed values with shape `(..., qty.shape)` for each `qty` key are returned.
                  The return `dict` also has a `coords` key with shape `(qty.shape, dim)`.
        """
        if not self.compression:
            raise ValueError(f'Compression is not supported for variable "{self.name}". Please specify a compression'
                             f' method for this variable.')
        if not self.compression.map_exists:
            raise ValueError(f'Compression map not computed yet for "{self.name}".')

        # Default field coordinates to the compression coordinates if they are not provided
        field_coords = values.pop('coords', coords)
        if field_coords is None:
            field_coords = self.compression.coords
        ret_dict = {'coords': field_coords}

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

    def reconstruct(self, values, coords=None):
        """Alias for `compress(reconstruct=True)`. See `compress` for more details."""
        return self.compress(values, coords=coords, reconstruct=True)

    def serialize(self, save_path: str | Path = '.') -> dict:
        """Convert a `Variable` to a `dict` with only standard Python types
        (i.e. convert custom objects like `dist` and `norm` to strings and save `compression` to a `.pkl`).

        :param save_path: the path to save the compression data to (defaults to current directory)
        :returns: the serialized `dict` of the `Variable` object
        """
        d = {}
        for key, value in self.__dict__.items():
            if value is not None and not key.startswith('_'):
                if key == 'domain':
                    d[key] = [str(v) for v in value] if isinstance(value, list) else str(value)
                elif key == 'distribution':
                    d[key] = str(value)
                elif key == 'norm':
                    d[key] = [str(transform) for transform in value]
                elif key == 'compression':
                    fname = f'{self.name}_compression.pkl'
                    d[key] = value.serialize(save_path=Path(save_path) / fname)
                else:
                    d[key] = value
        return d

    @classmethod
    def deserialize(cls, data: dict, search_paths: list[str | Path] = None) -> Variable:
        """Convert a `dict` to a `Variable` object. Let `pydantic` handle validation and conversion of fields.

        :param data: the `dict` to convert to a `Variable`
        :param search_paths: the paths to search for compression files (if necessary)
        :returns: the `Variable` object
        """
        if isinstance(data, Variable):
            return data
        elif isinstance(data, str):
            return cls(name=data)
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
    """Store `Variables` as `str(var) : Variable` in the order they were passed in. You can:

    - Initialize/update from a single `Variable` or a list of `Variables`
    - Get/set a `Variable` directly or by name via `my_vars[var]` or `my_vars[str(var)]` etc.
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

    def get_domains(self, norm: bool = True):
        """Get normalized variable domains (expand latent coefficient domains for field quantities). Assume a
        domain of `(0, 1)` for variables if their domain is not specified.

        :param norm: whether to normalize the domains using `Variable.norm` (useful for getting bds for surrogate);
                     latent coefficient domains do not get normalized
        :returns: a `dict` of variables to their normalized domains; field quantities return a domain for each
                  of their latent coefficients
        """
        domains = {}
        for var in self:
            var_domain = var.get_domain()
            if isinstance(var_domain, list):  # only field qtys return a list of domains, one for each latent coeff
                for i, domain in enumerate(var_domain):
                    domains[f'{var.name}{LATENT_STR_ID}{i}'] = domain
            elif var_domain is None:
                domains[var.name] = (0, 1)
            else:
                domains[var.name] = var.normalize(var_domain) if norm else var_domain
        return domains

    def get_pdfs(self, norm: bool = True):
        """Get callable pdfs for all variables (skipping field quantities for now)

        :param norm: whether values passed to the pdf functions are normalized and should be denormed first
                     before pdf evaluation (useful for surrogate construction where samples are gathered in the
                     normalized space)
        :returns: a `dict` of variables to callable pdf functions; field quantities are skipped.
        """
        def _get_pdf(var, norm):
            return lambda z: var.pdf(var.denormalize(z) if norm else z)

        pdf_fcns = {}
        for var in self:
            var_domain = var.get_domain()
            if isinstance(var_domain, list):  # only field qtys return a list of domains, one for each latent coeff
                # for i, domain in enumerate(var_domain):
                    # pdf_fcns[f'{var.name}{LATENT_STR_ID}{i}'] = var.latent_pdfs[i]  TODO: Implement latent pdfs
                pass
            else:
                pdf_fcns[var.name] = _get_pdf(var, norm)
        return pdf_fcns

    def update(self, data: list[Variable | str] | str | Variable | OrderedDict | dict = None, **kwargs):
        """Update from a list or dict of `Variable` objects, or from `key=value` pairs."""
        if data:
            if isinstance(data, OrderedDict | dict):
                for key, value in data.items():
                    self.__setitem__(key, value)
            else:
                data = [data] if not isinstance(data, list | tuple) else data
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
        if isinstance(value, str):
            value = Variable(name=value)
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
        """Convert to a list of `dict` objects for each `Variable` in the list.

        :param save_path: the path to save the compression data to (defaults to current directory)
        """
        return [var.serialize(save_path=save_path) for var in self.values()]

    @classmethod
    def merge(cls, *variable_lists) -> VariableList:
        """Merge multiple sets of variables into a single `VariableList` object.

        !!! Note
            Variables with the same name will be merged by keeping the one with the most information provided.

        :param variable_lists: the variables/lists to merge
        :returns: the merged `VariableList` object
        """
        merged_vars = cls()

        def _get_best_variable(var1, var2):
            var1_dict = {key: value for key, value in var1.__dict__.items() if value is not None}
            var2_dict = {key: value for key, value in var2.__dict__.items() if value is not None}
            return var1 if len(var1_dict) >= len(var2_dict) else var2

        for var_list in variable_lists:
            for var in cls(var_list):
                if var.name in merged_vars:
                    merged_vars[var.name] = _get_best_variable(merged_vars[var.name], var)
                else:
                    merged_vars[var.name] = var

        return merged_vars

    @classmethod
    def deserialize(cls, data: dict | list[dict], search_paths=None) -> VariableList:
        """Convert a `dict` or list of `dict` objects to a `VariableList` object. Let `pydantic` handle validation."""
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
