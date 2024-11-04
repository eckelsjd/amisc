"""Module for data transformation methods.

Includes:

- `Transform` — an abstract interface for specifying a transformation.
- `Linear` — a linear transformation $y=mx+b$.
- `Log` — a logarithmic transformation $y=\\log_b(x + \\mathrm{offset})$.
- `Minmax` — a min-max scaling transformation $x: (lb, ub) \\mapsto (lb_{norm}, ub_{norm})$.
- `Zscore` — a z-score normalization transformation $y=(x-\\mu)/\\sigma$.

Transform objects can be converted easily to/from strings for serialization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from amisc.utils import parse_function_string

__all__ = ['Transform', 'Linear', 'Log', 'Minmax', 'Zscore']


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
        """Return a list of `Transforms` given a list of string specifications. Available transformations are:

        - **linear** — $x_{norm} = mx + b$ specified as `linear(m, b)` or `linear(slope=m, offset=b)`. `m=1, b=0` if not
                       specified.
        - **log** — $x_{norm} = \\log_b(x)$ specified as `log` or `log10` for the natural or common logarithms. For a
                    different base, use `log(b)`. Optionally, specify `offset` for `log(x+offset)`.
        - **minmax** — $x_{norm} = \\frac{x - a}{b - a}(u - l) + l$ specified as `minmax(a, b, l, u)` or
                       `minmax(lb=a, ub=b, lb_norm=l, ub_norm=u)`. Scales `x` from the range `(a, b)` to `(l, u)`. By
                       default, `(a, b)` is the Variable's domain and `(l, u)` is `(0, 1)`. Use simply as `minmax`
                       to use all defaults.
        - **zscore** — $x_{norm} = \\frac{x - m}{s}$ specified as `zscore(m, s)` or `zscore(mu=m, std=s)`. If the
                       Variable is specified as `distribution=normal`, then `zscore` defaults to the Variable's
                       own `mu, std`.

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
            `Variable`, but this will throw a runtime error if `Variable.distribution` is not `Normal(mu, std)`.
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
        """Transform the given values `x`. This wrapper function handles the input type and tries to
         return the transformed values in the same type.

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
        """Abstract method that subclass `Transform` objects should implement.

        :param x: the values to transform
        :param inverse: whether to do the inverse transform instead
        :param transform_args: overrides `Transform.transform_args`
        :return: the transformed values
        """
        raise NotImplementedError


class Linear(Transform):
    """A Linear transform: $y=mx+b$.

    :ivar transform_args: `(m, b)` the slope and offset
    """
    def _transform(self, x, inverse=False, transform_args=None):
        slope, offset = transform_args or self.transform_args
        return (x - offset) / slope if inverse else slope * x + offset


class Log(Transform):
    """A Log transform: $y=\\log_b(x + \\mathrm{offset})$.

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
