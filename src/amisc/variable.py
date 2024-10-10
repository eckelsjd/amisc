"""Provides an object-oriented interface for model inputs/outputs and random variables.

Includes:

- `Variable` — an object that stores information about a variable and includes methods for sampling, pdf evaluation,
               normalization, compression, loading from file, etc.
- `VariableList` — a container for Variables that provides dict-like access of Variables by `name` along with normal
                   indexing and slicing

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
from amisc.distribution import Distribution, Normal
from amisc.serialize import Serializable
from amisc.transform import Transform, Minmax, Zscore
from amisc.utils import as_tuple, search_for_file, _get_yaml_path, _inspect_assignment
from amisc.typing import CompressionData

__all__ = ['Variable', 'VariableList']
_Transformation = Union[str, Transform, list[str | Transform]]  # something that can be converted to a Transform


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
      such as from the output of many simulation software programs. See the `Compression` classes.
    - Use `Variable.category` as an additional layer for using Variable's in different ways (e.g. set a "calibration"
      category for Bayesian inference).

    !!! Example
        ```python
        # Random variable
        temp = Variable(name='T', description='Temperature', units='K', dist='Uniform(280, 320)')
        samples = temp.sample(100)
        pdf = temp.pdf(samples)

        # Field quantity
        vel = Variable(name='u', description='Velocity', units='m/s', compression={'quantities': ['ux', 'uy', 'uz']})
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
    :ivar shape: the shape of the variable (e.g. for field quantities) -- empty for scalar variables (default)
    :ivar compression: specifies field quantities and links to relevant compression data
    :ivar dist: a string specifier of a probability distribution function (see the `Distribution` types)
    :ivar domain: the explicit domain bounds of the variable (limits of where you expect to use it)
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
    shape: Optional[str | tuple[int, ...]] = None
    compression: Optional[str | dict | Compression] = None
    dist: Optional[str | Distribution] = None
    domain: Optional[str | tuple[float, float]] = None
    norm: Optional[_Transformation] = None

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

    @field_validator('shape')
    @classmethod
    def _validate_shape(cls, shape) -> tuple[int, ...]:
        if shape is None:
            return ()
        return as_tuple(shape)

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
    def _validate_norm(cls, norm: _Transformation, info: ValidationInfo) -> list[Transform] | None:
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

    def update_domain(self, domain: tuple[float, float], transform: bool = False):
        """Update the domain of this variable.

        :param domain: the new domain to set
        :param transform: whether to update the domain in the transformed space instead
        """
        curr_domain = self.get_domain(transform=transform)
        curr_lb, curr_ub = curr_domain if curr_domain is not None else (None, None)
        lb, ub = domain
        lb = min(lb, curr_lb) if curr_lb is not None else lb
        ub = max(ub, curr_ub) if curr_ub is not None else ub
        self.domain = self.denormalize((lb, ub)) if transform else (lb, ub)

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

    def serialize(self, save_path: str | Path = '.') -> dict:
        """Convert a `Variable` to a `dict` with only standard Python types
        (i.e. convert custom objects like `dist` and `norm` to strings).
        """
        d = {}
        for key, value in self.__dict__.items():
            if value is not None and not key.startswith('_'):
                if key in ['domain', 'shape']:
                    if len(value) > 0:
                        d[key] = str(value)
                elif key == 'dist':
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
    def deserialize(cls, data: dict, search_paths=None) -> Variable:
        """Convert a `dict` to a `Variable` object. Let `pydantic` handle validation and conversion of fields."""
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
        return [var.serialize(save_path=save_path) for var in self.values()]

    @classmethod
    def merge(cls, *variable_lists) -> VariableList:
        """Merge multiple sets of variables into a single `VariableList` object.

        !!! Note
            Variables with the same name will be merged by keeping the one with the most information provided.
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
