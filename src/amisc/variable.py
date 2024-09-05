"""Provides an object-oriented interface for model inputs/outputs and random variables.

Includes:

- `Variable`: an object that stores information about a variable and includes methods for sampling, pdf evaluation,
              normalization, compression, loading from file, etc.
"""
import ast
import random
import string
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.interpolate import RBFInterpolator

Numeric = int | float | list | tuple | np.ndarray | np.generic


class Variable:
    """Object for storing information about variables and providing methods for pdf evaluation, sampling, etc.

    A simple variable object can be created with `var = Variable()`. All initialization options are optional and will
    be given good defaults. You should probably at the very least give a memorable `var_id` and a `domain`.

    With the `pyyaml` library installed, all Variable objects can be saved or loaded directly from a `.yml` file by
    using the `!Variable` yaml tag (which is loaded by default with `amisc`).

    - Use `Variable.dist` to specify sampling PDFs, such as for random variables.
    - Use `Variable.norm` to specify a transformed-space that is more amenable to surrogate construction
      (e.g. mapping to the range (0,1))
    - Use `Variable.field` to specify high-dimensional, coordinate-based field quantities, such as from the output of
      many simulation software programs
    - Use `Variable.category` as an additional layer for using Variable's in different ways (e.g. set a "calibration"
      category for Bayesian inference)

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

    :ivar var_id: an identifier for the variable, can compare variables directly with strings for indexing purposes
    :ivar domain: the explicit domain bounds of the variable (limits of where you expect to use it)
    :ivar nominal: a typical value for this variable (within `domain`)
    :ivar tex: latex format for the variable, i.e. r"$x_i$"
    :ivar dist: a specifier of a probability distribution function
    :ivar description: a lengthier description of the variable
    :ivar units: assumed units for the variable (if applicable)
    :ivar category: an additional descriptor for how this variable is used, e.g. calibration, operating, design, etc.
    :ivar norm: a specifier of a map to a transformed-space for surrogate construction
    :ivar field: specifies field quantities and links to relevant compression data

    :vartype var_id: str
    :vartype domain: tuple[float, float]
    :vartype nominal: float
    :vartype tex: str
    :vartype dist: str
    :vartype description: str
    :vartype units: str
    :vartype category: str
    :vartype norm: str
    :vartype field: dict
    """
    yaml_tag = u'!Variable'

    def __init__(self, var_id: str = None, domain: tuple | str = None, tex: str = None, nominal: float = None,
                 dist: str = None, description: str = None, units: str = None, category: str = None,
                 norm: str = None, field: dict | bool = None):
        self.tex = tex
        self.nominal = nominal
        self.dist = dist
        self.description = description
        self.units = units
        self.category = category
        self.norm = norm
        self.var_id = var_id or 'X_' + ''.join(random.choices(string.digits, k=3))
        if self.tex:
            self.tex = rf'{tex}' if tex.startswith('$') else rf'${tex}$'
        if field:
            if not isinstance(field, dict):
                field = dict()
            field['quantities'] = field.get('quantities', [])
            field['compress_method'] = field.get('compress_method', 'SVD')
            field['compress_datapath'] = field.get('compress_datapath', f'{self.var_id}_svd.h5')
            field['compress_args'] = field.get('compress_args', dict())
            field['interp_args'] = field.get('interp_args', dict())
        self.field = field

        # Make sure domain is specified in some way
        if dist := self.dist_from_string(dist):
            dist_name, dist_args = dist
            if dist_name == 'normal':
                mu, std = dist_args
                domain = domain or (mu - 3*std, mu + 3*std)
            elif dist_name == 'uniform':
                domain = domain or dist_args
        domain = domain or (0, 1)
        if isinstance(domain, str):
            domain = ast.literal_eval(domain.strip())
        self.domain = tuple(domain)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        self.update({key: value})

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
            return NotImplementedError(f'Cannot compare the "{self.var_id}" Variable to {other}.')

    def update(self, new_attrs: dict = None, **kwargs):
        """Update this variable with `new_attrs` or any specific attributes via `update(domain=...)` for example."""
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

    def get_nominal(self, norm: bool = False) -> float:
        """Return the nominal value of the variable. Defaults to the mean for a normal distribution or the
        center of the domain if `var.nominal` is not specified.

        :param norm: return the nominal value in normed space using `Variable.norm`
        """
        nominal = self.normalize(self.nominal) if norm else self.nominal
        if nominal is None:
            if dist := self.dist_from_string(self.dist):
                dist_name, dist_args = dist
                dist_args = self.normalize(dist_args) if norm else dist_args
                if dist_name == 'normal':
                    nominal = dist_args[0]  # the mean
                elif dist_name == 'uniform':
                    nominal = (dist_args[0] + dist_args[1]) / 2
            else:
                domain = self.get_domain(norm=norm)
                nominal = (domain[0] + domain[1]) / 2

        return nominal

    def get_domain(self, norm: bool = False) -> tuple:
        """Return a tuple of the defined domain of this variable.

        :param norm: return the domain of the normed space instead
        """
        return tuple(self.normalize(self.domain)) if norm else self.domain

    def sample_domain(self, shape: tuple | int, norm: bool = False) -> np.ndarray:
        """Return an array of the given `shape` for random samples over the domain of this variable.

        :param shape: the shape of samples to return
        :param norm: whether to sample in the normed space instead
        :returns: the random samples over the domain of the variable
        """
        if isinstance(shape, int):
            shape = (shape, )
        domain = self.get_domain(norm=norm)
        return np.random.rand(*shape) * (domain[1] - domain[0]) + domain[0]

    def pdf(self, x: np.ndarray, norm: bool = False) -> np.ndarray:
        """Compute the PDF of the Variable at the given `x` locations.

        !!! Note
            If `norm=True`, then the arguments of `self.dist` and `x` will both be normalized via `self.norm`
            _before_ computing the PDF. Note that this
            means if `x=Variable(dist=N(0, 1), norm=linear(m=2, b=2))`, then $x$ is not normally
            distributed as `N(0,1)`; rather, a new variable $y$ is distributed as $y\\sim\\mathcal{N}(2, 4)$ -- i.e.
            the `dist` parameters get transformed, not the distribution itself. `x.pdf(norm=True)` will then
            actually return the pdf of the transformed variable $y$. Or for example,
            a log-uniform variable can be obtained via `Variable(dist=U(1, 10), norm=log10)`.

        :param x: locations to compute the PDF at
        :param norm: whether to compute the PDF in normed-space instead
        :returns: the PDF evaluations at `x`
        """
        y = self.normalize(x) if norm else x
        if dist := self.dist_from_string(self.dist):
            dist_name, dist_args = dist
            if dist_name == 'normal':
                mu, std = self.normalize(dist_args) if norm else dist_args
                return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((y - mu) / std) ** 2)
            elif dist_name == 'uniform':
                lb, ub = self.normalize(dist_args) if norm else dist_args
                pdf = np.broadcast_to(1 / (ub - lb), y.shape).copy()
                pdf[np.where(y > ub)] = 0
                pdf[np.where(y < lb)] = 0
                return pdf
        else:
            return np.ones(x.shape)  # No pdf if no dist or anything else is specified

    def sample(self, shape: tuple | int, nominal: float = None, norm: bool = False) -> np.ndarray:
        """Draw samples from this Variable's distribution. Just returns the nominal value of the given shape if
        this Variable has no distribution.

        !!! Note
            If `norm=True`, then samples will be drawn in this Variable's normed-space. This would be
            used for example if sampling a log-uniform distributed variable via
            `x=Variable(dist=U(1, 10), norm=log); x.sample(norm=True)`.

        :param shape: the shape of the returned samples
        :param nominal: a nominal value to use if applicable (i.e. a center for relative, tolerance, or normal)
        :param norm: whether to sample in the Variable's normed space instead
        :returns: samples from the PDF of this Variable's distribution
        """
        if isinstance(shape, int):
            shape = (shape, )

        if dist := self.dist_from_string(self.dist):
            dist_name, dist_args = dist
            if dist_name == 'normal':
                mu, std = dist_args
                mu = nominal or mu
                if norm:
                    mu, std = self.normalize((mu, std))
                samples = np.random.randn(*shape) * std + mu
                return samples
            elif dist_name == 'uniform':
                return self.sample_domain(shape, norm=norm)
            elif dist_name in ['relative', 'tolerance']:
                nominal = (self.normalize(nominal) if norm else nominal) or self.get_nominal(norm=norm)
                if nominal is None:
                    raise ValueError(f'Cannot sample "{self.var_id}" with a {dist_name} distribution when no nominal'
                                     f'value is provided.')
                else:
                    tol = abs(dist_args[0]) if dist_name == 'tolerance' else abs(dist_args[0] * nominal)
                    return (np.random.rand(*shape) * (2*tol) - tol) + nominal
        else:
            # Variable's with no distribution
            nominal = (self.normalize(nominal) if norm else nominal) or self.get_nominal(norm=norm)
            if nominal is None:
                raise ValueError(f'Cannot sample "{self.var_id}" with no dist or nominal value specified.')
            else:
                return np.ones(shape) * nominal

    def normalize(self, values: Numeric, denorm: bool = False) -> Numeric | None:
        """Normalize `values` based on this Variable's `norm` method(s). Available `self.norm` methods are:

        - **linear**: $x_{norm} = mx + b$ specified as `linear(m, b)` or `linear(slope=m, offset=b)`. `b=0` if not
                      specified. Use `m=1/10` for fractional values (for example).
        - **log**: $x_{norm} = \\log_b(x)$ specified as `log` or `log10` for the natural or common logarithms. For a
                   different base, use `log(b)`. Optionally, specify `offset=1` for `log(x+1)`.
        - **minmax**: $x_{norm} = \\frac{x - a}{b - a}(u - l) + l$ specified as `minmax(a, b, l, u)` or
                      `minmax(lb=a, ub=b, lb_norm=l, ub_norm=u)`. Scales `x` from the range `(a, b)` to `(l, u)`. By
                      default, `(a, b)` is the Variable's domain and `(l, u)` is `(0, 1)`. Use simply as `minmax`
                      to use all defaults.
        - **zscore**: $x_{norm} = \\frac{x - m}{s}$ specified as `zscore(m, s)` or `zscore(mu=m, std=s)`. If the
                      Variable is specified as `dist=normal`, then `zscore` defaults to the Variable's own `mu, std`.

        !!! Note
            If this Variable's `self.norm` was specified as a list of norm methods, then each will be applied in
            sequence in the original order (and in reverse for `denorm=True`).

        :param values: the values to normalize
        :param denorm: whether to denormalize instead using the inverse of the original normalization method
        :returns: the normalized (or unnormalized) values
        """
        if not self.norm or values is None:
            return values
        values = np.atleast_1d(values)
        norm_methods = [self.norm] if isinstance(self.norm, str) else self.norm

        def _normalize_single(values, norm_string, denorm, domain, dist_args):
            norm_name, args, kwargs = Variable.parse_function_string(norm_string)
            if norm_name.lower() == 'linear':
                try:
                    slope = kwargs.get('slope', args[0] if args else 1)
                    slope = float(slope.split('/')[0]) / float(slope.split('/')[1]) if '/' in slope else float(slope)
                    offset = float(kwargs.get('offset', args[1] if len(args) > 1 else 0))
                    return (values - offset) / slope if denorm else slope * values + offset
                except Exception:
                    raise ValueError(f'Linear normalization string "{norm_string}" is not valid: Try "linear(m, b)".')
            elif norm_name.lower() in ['log', 'log10']:
                try:
                    log_base = kwargs.get('base', args[0] if args else ('e' if norm_name.lower() == 'log' else 10))
                    log_base = np.e if log_base == 'e' else float(log_base)
                    a = float(kwargs.get('offset', 0))
                    return np.exp(values * np.log(log_base)) - a if denorm else np.log(values + a) / np.log(log_base)
                except Exception:
                    raise ValueError(f'Log normalization string "{norm_string}" is not valid: Try "log(b)".')
            elif norm_name.lower() in ['minmax', 'maxabs']:
                try:
                    lb = float(kwargs.get('lb', args[0] if args else domain[0]))
                    ub = float(kwargs.get('ub', args[1] if args else domain[1]))
                    lb_norm = float(kwargs.get('lb_norm', args[2] if len(args) > 2 else 0))
                    ub_norm = float(kwargs.get('ub_norm', args[3] if len(args) > 2 else 1))
                    if denorm:
                        return (values - lb_norm) / (ub_norm - lb_norm) * (ub - lb) + lb
                    else:
                        return (values - lb) / (ub - lb) * (ub_norm - lb_norm) + lb_norm
                except Exception:
                    raise ValueError(f'Minmax normalization string "{norm_string}" is not valid: Try "minmax(lb, ub)".')
            elif norm_name.lower() in ['z', 'zscore']:
                try:
                    mu, std = None, None
                    if args:
                        mu, std = args[0], args[1]
                    elif kwargs:
                        mu = kwargs.get('mu', None)
                        std = kwargs.get('std', None)
                    else:
                        if dist := self.dist_from_string(self.dist):
                            dist_name, _ = dist
                            if dist_name == 'normal':
                                mu, std = dist_args
                            else:
                                raise ValueError(f'Z-score normalization not valid for variable "{self.var_id}" with '
                                                 f'distribution "{self.dist}" when mu and std are not provided.')
                    if mu and std:
                        mu, std = float(mu), float(std)
                        return values * std + mu if denorm else (values - mu) / std
                    else:
                        raise ValueError
                except Exception:
                    raise ValueError(f'Z-score normalization string "{norm_string}" is not valid: '
                                     f'Try "zscore(mu, std)".')
            else:
                raise NotImplementedError(f'Normalization method "{norm_name}" is not implemented.')

        domain, dist_args = self.domain, []
        if dist := self.dist_from_string(self.dist):
            _, dist_args = dist

        if denorm:
            # First, send domain and dist_args through the forward norm list (up until the last norm)
            hyperparams = [np.hstack((domain, dist_args))]
            for i, norm_string in enumerate(norm_methods):
                domain, dist_args = tuple(hyperparams[i][:2]), tuple(hyperparams[i][2:])
                hyperparams.append(_normalize_single(hyperparams[i], norm_string, False, domain, dist_args))

            # Now denormalize in reverse
            hp_idx = -2
            for norm_string in reversed(norm_methods):
                domain, dist_args = tuple(hyperparams[hp_idx][:2]), tuple(hyperparams[hp_idx][2:])
                values = _normalize_single(values, norm_string, denorm, domain, dist_args)
                hp_idx -= 1
        else:
            # Normalize values and hyperparams through the forward norm list
            hyperparams = np.hstack((domain, dist_args))
            for norm_string in norm_methods:
                domain, dist_args = tuple(hyperparams[:2]), tuple(hyperparams[2:])
                values = _normalize_single(values, norm_string, denorm, domain, dist_args)
                hyperparams = _normalize_single(hyperparams, norm_string, denorm, domain, dist_args)
        return values

    def denormalize(self, values):
        """Alias for `normalize(denorm=True)`"""
        return self.normalize(values, denorm=True)

    def compress(self, values: dict, reconstruct: bool = False, interp_kwargs: dict = None) -> dict:
        """Compress or reconstruct field quantity values using this Variable's compression data.

        !!! Note
            If only one field quantity is associated with this variable, then `len(field[quantities])=0`. In this case,
            specify `values` as `dict(coord=..., var_id=...)` for this Variable's `var_id`. If `coord` is not specified,
            then this assumes the locations are the same as the reconstruction data (and skips interpolation).

        !!! Info
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
        :returns: the compressed values with key `latent` and shape `(..., reduced_rank)`; if `reconstruct=True`,
                  then the reconstructed values with shape `(..., qty.shape)` for each `qty` key are returned.
                  The return `dict` also has a `coord` key with shape `(qty.shape, dim)`.
        """
        field = self.field
        if not field:
            raise ValueError(f'Compression is not supported for the non-field variable "{self.var_id}".')
        method = field['compress_method']
        datapath = Path(field['compress_datapath'])
        interp_kwargs = interp_kwargs or field['interp_args']
        if not datapath.exists():
            raise ValueError(f'The compression datapath "{datapath}" for variable "{self.var_id}" does not exist.')

        nqoi = max(1, len(field['quantities']))
        qty_list = field['quantities'] if nqoi > 1 else [self.var_id]

        if method.lower() == 'svd':
            if datapath.suffix not in ['.h5', '.hdf5']:
                raise ValueError(f'Compression file type "{datapath.suffix}" is not supported for SVD. '
                                 f'Only [.h5, .hdf5].')
            with h5py.File(datapath, 'r') as fd:
                projection_matrix = fd['projection_matrix'][:]   # (dof, rank)
                rec_coords = fd['coord'][:]                      # (num_pts, dim)
                num_pts, dim = rec_coords.shape                  # Also num_pts = dof / nqoi
                reconstruct_func = lambda latent: np.squeeze(projection_matrix @ latent[..., np.newaxis], axis=-1)
                compress_func = lambda states: np.squeeze(projection_matrix.T @ states[..., np.newaxis], axis=-1)
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
        field_shape = field_coords.shape[:-1]
        field_coords = field_coords.reshape(-1, dim)    # (N, dim)

        # Don't need to interpolate if passed-in coords are same as reconstruction coords
        no_interp = no_interp or (field_coords.shape == rec_coords.shape and np.allclose(field_coords, rec_coords))
        ret_dict = {'coord': values.get('coord', rec_coords)}

        # For reconstruction: decompress -> denormalize -> interpolate
        if reconstruct:
            try:
                latent = np.atleast_1d(values['latent'])                                # (..., rank)
            except KeyError:
                raise ValueError('Must pass values["latent"] in for reconstruction.')
            states = reconstruct_func(latent)                                           # (..., dof)
            states = self.denormalize(states).reshape((*states.shape[:-1], num_pts, nqoi))

            for i, qoi in enumerate(qty_list):
                if no_interp:
                    ret_dict[qoi] = states[..., i]
                else:
                    reshaped_states = states[..., i].reshape(-1, num_pts).T                     # (num_pts, N...)
                    rbf = RBFInterpolator(rec_coords, reshaped_states, **interp_kwargs)
                    field_vals = rbf(field_coords).T.reshape(*states.shape[:-2], *field_shape)  # (..., field_shape)
                    ret_dict[qoi] = field_vals
                # field_vals = states[..., i] if no_interp else (
                #     griddata(rec_coords, states[..., i], field_coords, method=interp_method)
                # )
                # ret_dict[qoi] = field_vals.reshape((*states.shape[:-1], *field_shape))  # (..., field_shape)

        # For compression: interpolate -> normalize -> compress
        else:
            states = None
            for i, qoi in enumerate(qty_list):
                if states is None:
                    states = np.empty((*values[qoi].shape[:-len(field_shape)], num_pts, nqoi))
                field_vals = values[qoi].reshape((*states.shape[:-2], -1))              # (..., N)
                if no_interp:
                    states[..., i] = field_vals
                else:
                    reshaped_field = field_vals.reshape(-1, field_vals.shape[-1]).T     # (N, ...)
                    rbf = RBFInterpolator(field_coords, reshaped_field, **interp_kwargs)
                    states[..., i] = rbf(rec_coords).T.reshape(*states.shape[:-2], num_pts)
                # states[..., i] = field_vals if no_interp else (
                #     griddata(field_coords, field_vals, rec_coords, method=interp_method)
                # )
            states = self.normalize(np.reshape(states, (*states.shape[:-2], -1)))       # (..., dof)
            latent = compress_func(states)                                              # (..., rank)
            ret_dict['latent'] = latent

        return ret_dict

    def reconstruct(self, values, **kwargs):
        """Alias for `compress(reconstruct=True)`"""
        return self.compress(values, reconstruct=True, **kwargs)

    @staticmethod
    def parse_function_string(func_string):
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
        except Exception:
            raise ValueError(f'Function string "{func_string}" is not valid.')

    @staticmethod
    def dist_from_string(dist_string: str) -> (str, tuple):
        """Convert a string to a distribution.

        :param dist_string: specifies a PDF or distribution. Can be `Normal(mu, std)`, `Uniform(lb, ub)`,
                            `Relative(pct)`, or `Tolerance(tol)`. The shorthands `N(0, 1)`, `U(0, 1)`, `rel(5%)`, or
                            `tol(1)` are also accepted.
        :returns: `dist_name`, `dist_args` - the distribution name and arguments; `(mu, std)` for `normal`, `(lb, ub)`
                                             for `uniform`, `(pct,)` for `relative`, `(tol,)` for `tolerance`
        """
        if not dist_string:
            return None

        dist_name, args, kwargs = Variable.parse_function_string(dist_string)
        if dist_name in ['N', 'Normal', 'normal']:
            # Normal distribution like N(0, 1)
            try:
                mu = float(kwargs.get('mu', args[0]))
                std = float(kwargs.get('std', args[1]))
                return 'normal', (mu, std)
            except Exception:
                raise ValueError(f'Normal distribution string "{dist_string}" is not valid: Try N(0, 1).')
        elif dist_name in ['U', 'Uniform', 'uniform']:
            # Uniform distribution like U(0, 1)
            try:
                lb = float(kwargs.get('lb', args[0]))
                ub = float(kwargs.get('ub', args[1]))
                return 'uniform', (lb, ub)
            except Exception:
                raise ValueError(f'Uniform distribution string "{dist_string}" is not valid: Try U(0, 1).')
        elif dist_name in ['R', 'Relative', 'relative', 'rel']:
            # Relative uniform distribution like rel(+-5%)
            try:
                pct = float(kwargs.get('pct', args[0].split(r'%')[0].strip()))
                return 'relative', (pct / 100,)
            except Exception:
                raise ValueError(f'Relative distribution string "{dist_string}" is not valid: Try rel(5%).')
        elif dist_name in ['T', 'Tolerance', 'tolerance', 'tol']:
            # Uniform distribution within a tolerance like tol(+-1)
            try:
                tol = float(kwargs.get('tol', args[0]))
                return 'tolerance', (tol,)
            except Exception:
                raise ValueError(f'Tolerance distribution string "{dist_string}" is not valid: Try tol(1).')
        else:
            raise NotImplementedError(f'The distribution "{dist_string}" is not recognized.')

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, data) -> yaml.MappingNode:
        """Convert a single `Variable` object (`data`) to a yaml MappingNode (i.e. a dict)."""
        instance_variables = {key: value for key, value in data.__dict__.items() if value is not None}
        return dumper.represent_mapping(Variable.yaml_tag, instance_variables)

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the !Variable tag in yaml to a single `Variable` object (or a list of `Variables`)."""
        if isinstance(node, yaml.SequenceNode):
            return [Variable(**variable_dict) for variable_dict in loader.construct_sequence(node, deep=True)]
        elif isinstance(node, yaml.MappingNode):
            return Variable(**loader.construct_mapping(node))
        else:
            raise NotImplementedError(f'The "{Variable.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')
