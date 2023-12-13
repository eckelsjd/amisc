"""`utils.py`

Provides some basic utilities for the package.

Includes
--------
- `load_variables`: convenience function for loading RVs from a .json config file
- `get_logger`: logging utility with nice formatting
- `ax_default`: plotting utility with nice formatting
- `approx_hess`: finite difference approximation of the Hessian
- `batch_normal_sample`: helper function to sample from arbitrarily-sized Gaussian distribution(s)
- `ndscatter`: plotting utility for n-dimensional data
"""
import json
from pathlib import Path
import logging
import sys

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import StrMethodFormatter, AutoLocator, FuncFormatter
import scipy.stats as st
import numpy as np
from numpy.linalg.linalg import LinAlgError

from amisc.rv import BaseRV, UniformRV, NormalRV, ScalarRV


LOG_FORMATTER = logging.Formatter("%(asctime)s \u2014 [%(levelname)s] \u2014 %(name)-20s \u2014 %(message)s")


def load_variables(variables: list[str], file: Path | str) -> list[BaseRV]:
    """Load a list of BaseRV objects from a variables json `file`.

    :param variables: a list of str ids for variables to find in `file`
    :param file: json file to search for variable definitions
    :returns rvs: a list of corresponding `BaseRV` objects
    """
    with open(Path(file), 'r') as fd:
        data = json.load(fd)

    rvs = []
    keys = ['id', 'tex', 'description', 'units', 'param_type', 'nominal', 'domain']
    for str_id in variables:
        if str_id in data:
            var_info = data.get(str_id)
            kwargs = {key: var_info.get(key) for key in keys if var_info.get(key)}
            match var_info.get('rv_type', 'none'):
                case 'uniform_bds':
                    bds = var_info.get('rv_params')
                    rvs.append(UniformRV(bds[0], bds[1], **kwargs))
                case 'uniform_pct':
                    rvs.append(UniformRV(var_info.get('rv_params'), 'pct', **kwargs))
                case 'uniform_tol':
                    rvs.append(UniformRV(var_info.get('rv_params'), 'tol', **kwargs))
                case 'normal':
                    mu, std = var_info.get('rv_params')
                    rvs.append(NormalRV(mu, std, **kwargs))
                case 'none':
                    # Make a plain stand-in scalar RV object (no uncertainty)
                    rvs.append(ScalarRV(**kwargs))
                case other:
                    raise NotImplementedError(f'RV type "{var_info.get("rv_type")}" is not known.')
        else:
            raise ValueError(f'You have requested the variable {str_id}, but it was not found in {file}. '
                             f'Please add a definition of {str_id} to {file} or construct it on your own.')

    return rvs


def get_logger(name: str, stdout=True, log_file: str | Path = None) -> logging.Logger:
    """Return a file/stdout logger with the given name.

    :param name: the name of the logger to return
    :param stdout: whether to add a stdout handler to the logger
    :param log_file: add file logging to this file (optional)
    :returns: the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger


def approx_hess(func: callable, theta: np.ndarray, pert=0.01) -> np.ndarray:
    """Approximate Hessian of `func` at a specified `theta` location.

    :param func: expects to be called as `func(theta) -> (..., y_dim)`, where `y_dim=1` (scalar funcs only)
    :param theta: `(..., theta_dim)`, points to linearize model about
    :param pert: perturbation for approximate partial derivatives
    :returns H: `(..., theta_dim, theta_dim)`, the approximate Hessian `(theta_dim, theta_dim)` at all locations (...)
    """
    theta = np.atleast_1d(theta)
    shape = theta.shape[:-1]                # (*)
    theta_dim = theta.shape[-1]             # Number of parameters
    dtheta = pert * theta

    # Return the Hessians (..., theta_dim, theta_dim)
    H = np.empty(shape + (theta_dim, theta_dim))

    for i in range(theta_dim):
        for j in range(i, theta_dim):
            # Allocate space at 4 grid points (n1=-1, p1=+1)
            theta_n1_n1 = np.copy(theta)
            theta_p1_p1 = np.copy(theta)
            theta_n1_p1 = np.copy(theta)
            theta_p1_n1 = np.copy(theta)

            # Perturbations to theta in each direction
            theta_n1_n1[..., i] -= dtheta[..., i]
            theta_n1_n1[..., j] -= dtheta[..., j]
            f_n1_n1 = func(theta_n1_n1)

            theta_p1_p1[..., i] += dtheta[..., i]
            theta_p1_p1[..., j] += dtheta[..., j]
            f_p1_p1 = func(theta_p1_p1)

            theta_n1_p1[..., i] -= dtheta[..., i]
            theta_n1_p1[..., j] += dtheta[..., j]
            f_n1_p1 = func(theta_n1_p1)

            theta_p1_n1[..., i] += dtheta[..., i]
            theta_p1_n1[..., j] -= dtheta[..., j]
            f_p1_n1 = func(theta_p1_n1)

            res = (f_n1_n1 + f_p1_p1 - f_n1_p1 - f_p1_n1) / np.expand_dims(4 * dtheta[..., i] * dtheta[..., j],
                                                                           axis=-1)

            # Hessian only computed for scalar functions, y_dim=1 on last axis
            H[..., i, j] = np.squeeze(res, axis=-1)
            H[..., j, i] = np.squeeze(res, axis=-1)

    return H


def batch_normal_sample(mean: np.ndarray | float, cov: np.ndarray | float, size: tuple | int = ()) -> np.ndarray:
    """Batch sample multivariate normal distributions (pretty much however you want).

    :param mean: `(..., dim)`, expected values, where dim is the random variable dimension
    :param cov: `(..., dim, dim)`, covariance matrices
    :param size: shape of additional samples
    :returns samples: `(*size, ..., dim)`, samples from multivariate distributions
    """
    mean = np.atleast_1d(mean)
    cov = np.atleast_2d(cov)

    if isinstance(size, int):
        size = (size, )
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    x_normal = np.random.standard_normal((*shape, 1)).astype(np.float32)
    samples = np.squeeze(np.linalg.cholesky(cov) @ x_normal, axis=-1) + mean
    return samples


def ax_default(ax: plt.Axes, xlabel='', ylabel='', legend=True, cmap='tab10'):
    """Nice default formatting for plotting X-Y data.

    :param ax: the axes to apply these settings to
    :param xlabel: the xlabel to set for `ax`
    :param ylabel: the ylabel to set for `ax`
    :param legend: whether to show a legend
    :param cmap: colormap to use for cycling
    """
    plt.rcParams["axes.prop_cycle"] = get_cycle(cmap)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    if legend:
        leg = ax.legend(fancybox=True)
        frame = leg.get_frame()
        frame.set_edgecolor('k')


def get_cycle(cmap: str | matplotlib.colors.Colormap, num_colors: int = None):
    """Get a color cycler for plotting.

    :param cmap: a string specifier of a matplotlib colormap (or a colormap instance)
    :param num_colors: the number of colors to cycle through
    """
    use_index = False
    if isinstance(cmap, str):
        use_index = cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
                             'tab10', 'tab20', 'tab20b', 'tab20c']
        cmap = matplotlib.cm.get_cmap(cmap)
    if num_colors is None:
        num_colors = cmap.N
    if cmap.N > 100:
        use_index = False
    elif isinstance(cmap, LinearSegmentedColormap):
        use_index = False
    elif isinstance(cmap, ListedColormap):
        use_index = True
    if use_index:
        ind = np.arange(int(num_colors)) % cmap.N
        return cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, num_colors))
        return cycler("color", colors)


def ndscatter(samples: np.ndarray, labels: list[str] = None, tick_fmts: list[str] = None, plot='scatter',
              cmap='viridis', bins=20, z: np.ndarray = None, cb_label=None, cb_norm='linear', subplot_size=3):
    """Triangle scatter plots of n-dimensional samples.

    !!! Warning
        Best for `dim < 10`. You can shrink the `subplot_size` to assist graphics loading time.

    :param samples: `(N, dim)` samples to plot
    :param labels: list of axis labels of length `dim`
    :param tick_fmts: list of str.format() specifiers for ticks, e.g `['{x: ^10.2f}', ...]`, of length `dim`
    :param plot: 'hist' for 2d hist plot, 'kde' for kernel density estimation, or 'scatter' (default)
    :param cmap: the matplotlib string specifier of a colormap
    :param bins: number of bins in each dimension for histogram marginals
    :param z: `(N,)` a performance metric corresponding to `samples`, used to color code the scatter plot if provided
    :param cb_label: label for color bar (if `z` is provided)
    :param cb_norm: `str` or `plt.colors.Normalize`, normalization method for plotting `z` on scatter plot
    :param subplot_size: size in inches of a single 2d marginal subplot
    :returns fig, axs: the `plt` Figure and Axes objects, (returns an additional `cb_fig, cb_ax` if `z` is specified)
    """
    N, dim = samples.shape
    x_min = np.min(samples, axis=0)
    x_max = np.max(samples, axis=0)
    if labels is None:
        labels = [f"x{i}" for i in range(dim)]
    if z is None:
        z = np.zeros(N)
    if cb_label is None:
        cb_label = 'Performance metric'

    def tick_format_func(value, pos):
        if value > 1:
            return f'{value:.2f}'
        if value > 0.01:
            return f'{value:.4f}'
        if value < 0.01:
            return f'{value:.2E}'
    default_ticks = FuncFormatter(tick_format_func)
    # if tick_fmts is None:
    #     tick_fmts = ['{x:.2G}' for i in range(dim)]

    # Set up triangle plot formatting
    fig, axs = plt.subplots(dim, dim, sharex='col', sharey='row')
    for i in range(dim):
        for j in range(dim):
            ax = axs[i, j]
            if i == j:                      # 1d marginals on diagonal
                # ax.get_shared_y_axes().remove(ax)
                ax._shared_axes['y'].remove(ax)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if i == 0:
                    ax.get_yaxis().set_ticks([])
            if j > i:                       # Clear the upper triangle
                ax.axis('off')
            if i == dim - 1:                # Bottom row
                ax.set_xlabel(labels[j])
                ax.xaxis.set_major_locator(AutoLocator())
                formatter = StrMethodFormatter(tick_fmts[j]) if tick_fmts is not None else default_ticks
                ax.xaxis.set_major_formatter(formatter)
            if j == 0 and i > 0:            # Left column
                ax.set_ylabel(labels[i])
                ax.yaxis.set_major_locator(AutoLocator())
                formatter = StrMethodFormatter(tick_fmts[i]) if tick_fmts is not None else default_ticks
                ax.yaxis.set_major_formatter(formatter)

    # Plot marginals
    for i in range(dim):
        for j in range(dim):
            ax = axs[i, j]
            if i == j:                      # 1d marginals (on diagonal)
                c = plt.get_cmap(cmap)(0)
                if plot == 'kde':
                    kernel = st.gaussian_kde(samples[:, i])
                    x = np.linspace(x_min[i], x_max[i], 1000)
                    ax.fill_between(x, y1=kernel(x), y2=0, lw=0, alpha=0.3, facecolor=c)
                    ax.plot(x, kernel(x), ls='-', c=c, lw=1.5)
                else:
                    ax.hist(samples[:, i], edgecolor='black', color=c, density=True, alpha=0.5,
                            linewidth=1.2, bins='auto')
            if j < i:                       # 2d marginals (lower triangle)
                ax.set_xlim([x_min[j], x_max[j]])
                ax.set_ylim([x_min[i], x_max[i]])
                if plot == 'scatter':
                    sc = ax.scatter(samples[:, j], samples[:, i], s=1.5, c=z, cmap=cmap, norm=cb_norm)
                elif plot == 'hist':
                    ax.hist2d(samples[:, j], samples[:, i], bins=bins, density=True, cmap=cmap)
                elif plot == 'kde':
                    kernel = st.gaussian_kde(samples[:, [j, i]].T)
                    xg, yg = np.meshgrid(np.linspace(x_min[j], x_max[j], 60), np.linspace(x_min[i], x_max[i], 60))
                    x = np.vstack([xg.ravel(), yg.ravel()])
                    zg = np.reshape(kernel(x), xg.shape)
                    ax.contourf(xg, yg, zg, 5, cmap=cmap, alpha=0.9)
                    ax.contour(xg, yg, zg, 5, colors='k', linewidths=1.5)
                else:
                    raise NotImplementedError('This plot type is not known. plot=["hist", "kde", "scatter"]')

    fig.set_size_inches(subplot_size * dim, subplot_size * dim)
    fig.tight_layout()

    # Plot colorbar in standalone figure
    if np.max(z) > 0 and plot == 'scatter':
        cb_fig, cb_ax = plt.subplots(figsize=(1.5, 6))
        cb_fig.subplots_adjust(right=0.7)
        cb_fig.colorbar(sc, cax=cb_ax, orientation='vertical', label=cb_label)
        cb_fig.tight_layout()
        return fig, axs, cb_fig, cb_ax

    return fig, axs
