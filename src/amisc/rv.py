"""`rv.py`

Provides small classes for random variables.

Includes
--------
- `BaseRV`: Abstract wrapper class of a random variable
- `UniformRV`: a uniformly-distributed random variable
- `NormalRV`: a normally-distributed random variable
- `ScalarRV`: a stand-in class for a variable with no uncertainty or pdf
- `LogUniformRV`: base 10 log-uniform
- `LogNormalRV`: base 10 log-normal
"""
from abc import ABC, abstractmethod
import random
import string

import numpy as np


class BaseRV(ABC):
    """Small wrapper class similar to `scipy.stats` random variables (RVs).

    :ivar id: an identifier for the variable
    :ivar bds: the explicit domain bounds of the variable (limits of where you expect to use it)
    :ivar nominal: a typical value for this variable (within `bds`)
    :ivar tex: latex format for the random variable, i.e. r"$x_i$"
    :ivar description: a lengthier description of the variable
    :ivar units: assumed units for the variable (if applicable)
    :ivar param_type: an additional descriptor for how this rv is used, e.g. calibration, operating, design, etc.

    :vartype id: str
    :vartype bds: tuple[float, float]
    :vartype nominal: float
    :vartype tex: str
    :vartype description: str
    :vartype units: str
    :vartype param_type: str
    """

    def __init__(self, id='', *, tex='', description='Random variable', units='-',
                 param_type='calibration', nominal=1, domain=(0, 1)):
        """Child classes must implement `sample` and `pdf` methods."""
        self.bds = tuple(domain)
        self.nominal = nominal
        self.id = id if id != '' else 'X_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        self.is_custom_id = id != ''  # Whether a custom id was assigned to this variable
        self.tex = tex
        self.description = description
        self.units = units
        self.param_type = param_type

    def __repr__(self):
        return r'{}'.format(f"{self.id} - {self.description} ({self.units})")

    def __str__(self):
        return self.id

    def __eq__(self, other):
        """Consider two RVs equal if they share the same string id.

        Also returns true when checking if this RV is equal to a string id by itself.
        """
        if isinstance(other, BaseRV):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.id)

    def to_tex(self, units=False, symbol=True):
        """Return a raw string that is well-formatted for plotting (with tex).

        :param units: whether to include the units in the string
        :param symbol: just latex symbol if true, otherwise the full description
        """
        s = self.tex if symbol else self.description
        if s == '':
            s = str(self)
        return r'{} [{}]'.format(s, self.units) if units else r'{}'.format(s)

    def bounds(self):
        """Return a tuple of the defined domain of this RV."""
        return self.bds

    def update_bounds(self, lb, ub):
        """Update the defined domain of this RV to `(lb, ub)`."""
        self.bds = (lb, ub)

    def sample_domain(self, shape: tuple | int) -> np.ndarray:
        """Return an array of the given `shape` for random samples over the domain of this RV.

        :param shape: the shape of samples to return
        :returns samples: random samples over the domain of the random variable
        """
        if isinstance(shape, int):
            shape = (shape, )
        return np.random.rand(*shape) * (self.bds[1] - self.bds[0]) + self.bds[0]

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the PDF of the RV at the given `x` locations.

        :param x: locations to compute the PDF at
        :returns f: the PDF evaluations at `x`
        """
        pass

    @abstractmethod
    def sample(self, shape: tuple | int, nominal: float = None) -> np.ndarray:
        """Draw samples from the PDF.

        :param shape: the shape of the returned samples
        :param nominal: a nominal value to use if applicable (i.e. a center for relative Uniform or Normal)
        :returns: samples from the PDF of this random variable
        """
        pass


class ScalarRV(BaseRV):
    """A stand-in variable with no uncertainty/pdf, just scalars."""

    def pdf(self, x):
        return np.ones(x.shape)

    def sample(self, shape, nominal=None):
        if isinstance(shape, int):
            shape = (shape, )
        if nominal is not None:
            return np.ones(shape)*nominal
        else:
            return self.sample_domain(shape)


class UniformRV(BaseRV):
    """A uniformly-distributed random variable.

    Can be uniformly distributed in one of three ways: between global bounds, relative within a percent, or relative
    within a set absolute tolerance.

    :ivar type: specifies the type of uniform distribution, either 'bds', 'pct', or 'tol' as described above
    :ivar value: the absolute tolerance or percent uncertainty if type is 'tol' or 'pct'

    :vartype type: str
    :vartype value: float
    """

    def __init__(self, arg1: float, arg2: float | str, id='', **kwargs):
        """Construct a uniformly-distributed random variable.

        :param arg1: lower bound if specifying U(lb, ub), otherwise a tol or pct if specifying U(+/- tol/pct)
        :param arg2: upper bound if specifying U(lb, ub), otherwise a str of either 'tol' or 'pct'
        """
        domain = kwargs.get('domain', None)
        if isinstance(arg2, str):
            self.value = arg1
            self.type = arg2
        else:
            self.value = None
            self.type = 'bds'
        if self.type == 'bds':
            domain = (arg1, arg2) if domain is None else tuple(domain)     # This means domain overrides (arg1, arg2)
        else:
            domain = (0, 1) if domain is None else tuple(domain)
        kwargs['domain'] = domain
        super().__init__(id, **kwargs)

        # Set default nominal value as middle of the domain if not specified
        if kwargs.get('nominal', None) is None:
            self.nominal = (self.bds[1] + self.bds[0]) / 2

    def __str__(self):
        return self.id if self.is_custom_id else f'U({self.bds[0]}, {self.bds[1]})'

    def get_uniform_bounds(self, nominal: float = None) -> tuple[float, float]:
        """Return the correct set of bounds based on type of uniform distribution.

        :param nominal: the center value for relative uniform distributions
        :returns: the uniform bounds
        """
        match self.type:
            case 'bds':
                return self.bds
            case 'pct':
                if nominal is None:
                    return self.bds
                return nominal * (1 - self.value), nominal * (1 + self.value)
            case 'tol':
                if nominal is None:
                    return self.bds
                return nominal - self.value, nominal + self.value
            case other:
                raise NotImplementedError(f'self.type = {self.type} not known. Choose from ["pct, "tol", "bds"]')

    def pdf(self, x: np.ndarray, nominal: float = None) -> np.ndarray:
        """Compute the pdf for a uniform distribution.

        :param x: locations to compute the pdf at
        :param nominal: center location for relative uniform rvs
        :returns: the evaluated PDF at `x`
        """
        bds = self.get_uniform_bounds(nominal)
        den = bds[1] - bds[0]
        den = 1 if np.isclose(den, 0) else den
        y = np.broadcast_to(1 / den, x.shape).copy()
        y[np.where(x > bds[1])] = 0
        y[np.where(x < bds[0])] = 0
        return y

    def sample(self, shape, nominal=None):
        if isinstance(shape, int):
            shape = (shape, )
        bds = self.get_uniform_bounds(nominal)
        return np.random.rand(*shape) * (bds[1] - bds[0]) + bds[0]


class LogUniformRV(BaseRV):
    """A base 10 log-uniform distributed random variable, only supports absolute bounds."""

    def __init__(self, log10_a: float, log10_b: float, id='', **kwargs):
        """Construct the log-uniform random variable.

        :param log10_a: the lower bound in log10 space
        :param log10_b: the upper bound in log10 space
        """
        super().__init__(id, **kwargs)
        self.bds = (10**log10_a, 10**log10_b)

    def __str__(self):
        return self.id if self.is_custom_id else f'LU({np.log10(self.bds[0]): .2f}, {np.log10(self.bds[1]): .2f})'

    def pdf(self, x):
        return np.log10(np.e) / (x * (np.log10(self.bds[1]) - np.log10(self.bds[0])))

    def sample(self, shape, nominal=None):
        if isinstance(shape, int):
            shape = (shape, )
        lb = np.log10(self.bds[0])
        ub = np.log10(self.bds[1])
        return 10 ** (np.random.rand(*shape) * (ub - lb) + lb)


class LogNormalRV(BaseRV):
    """A base 10 log-normal distributed random variable.

    :ivar mu: the center of the log-normal distribution
    :ivar std: the standard deviation of the log-normal distribution

    :vartype mu: float
    :vartype std: float
    """

    def __init__(self, mu: float, std: float, id='', **kwargs):
        """Construct the RV with the mean and std of the underlying distribution,
        i.e. $\\log_{10}(x) \\sim N(\\mu, \\sigma)$.

        :param mu: the center of the log-normal distribution
        :param std: the standard deviation of the log-normal distribution
        """
        domain = kwargs.get('domain', None)
        if domain is None:
            domain = (10 ** (mu - 3*std), 10 ** (mu + 3*std))   # Use a default domain of +- 3std
        kwargs['domain'] = domain
        super().__init__(id, **kwargs)
        self.std = std
        self.mu = mu

    def recenter(self, mu: float, std: float = None):
        """Move the center of the distribution to `mu` with standard deviation `std` (optional)

        :param mu: the new center of the distribution
        :param std: (optional) new standard deviation
        """
        self.mu = mu
        if std is not None:
            self.std = std

    def __str__(self):
        return self.id if self.is_custom_id else f'LN_10({self.mu}, {self.std})'

    def pdf(self, x):
        return (np.log10(np.e) / (x * self.std * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((np.log10(x) - self.mu) / self.std) ** 2)

    def sample(self, shape, nominal=None):
        if isinstance(shape, int):
            shape = (shape, )
        scale = np.log10(np.e)
        center = self.mu if nominal is None else nominal
        return np.random.lognormal(mean=(1 / scale) * center, sigma=(1 / scale) * self.std, size=shape)
        # return 10 ** (np.random.randn(*size)*self.std + center)  # Alternatively


class NormalRV(BaseRV):
    """A normally-distributed random variable.

    :ivar mu: float, the mean of the normal distribution
    :ivar std: float, the standard deviation of the normal distribution

    :vartype mu: float
    :vartype std: float
    """

    def __init__(self, mu, std, id='', **kwargs):
        domain = kwargs.get('domain', None)
        if domain is None:
            domain = (mu - 2.5*std, mu + 2.5*std)   # Use a default domain of +- 2.5std
        kwargs['domain'] = domain
        super().__init__(id, **kwargs)
        self.mu = mu
        self.std = std

        # Set default nominal value as the provided mean
        if kwargs.get('nominal', None) is None:
            self.nominal = mu

    def recenter(self, mu: float, std: float =None):
        """Move the center of the distribution to `mu` with standard deviation `std` (optional)

        :param mu: the new center of the distribution
        :param std: (optional) new standard deviation
        """
        self.mu = mu
        if std is not None:
            self.std = std

    def __str__(self):
        return self.id if self.is_custom_id else f'N({self.mu}, {self.std})'

    def pdf(self, x):
        return (1 / (np.sqrt(2 * np.pi) * self.std)) * np.exp(-0.5 * ((x - self.mu) / self.std) ** 2)

    def sample(self, shape, nominal=None):
        if isinstance(shape, int):
            shape = (shape, )
        center = self.mu if nominal is None else nominal
        return np.random.randn(*shape) * self.std + center
