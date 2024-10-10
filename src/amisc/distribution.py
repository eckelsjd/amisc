"""Module for probability distribution functions (PDFs).

Includes:

- `Distribution` — an abstract interface for specifying a PDF.
- `Uniform` — a uniform distribution.
- `Normal` — a normal distribution.
- `Relative` — a relative distribution (i.e. uniform within a percentage of a nominal value).
- `Tolerance` — a tolerance distribution (i.e. uniform within a tolerance of a nominal value).

Distribution objects can be converted easily to/from strings for serialization.
"""
from __future__ import annotations

from abc import abstractmethod, ABC

import numpy as np

from amisc.utils import parse_function_string

__all__ = ['Distribution', 'Uniform', 'Normal', 'Relative', 'Tolerance']


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
