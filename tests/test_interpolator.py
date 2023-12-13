import numpy as np
import matplotlib.pyplot as plt

from amisc.interpolator import LagrangeInterpolator
from amisc.rv import UniformRV
from amisc.utils import ax_default
from amisc.examples.models import tanh_func, nonlinear_wave


def test_tensor_product_1d(plots=False):
    beta = (3,)
    x_var = UniformRV(0, 1)
    x_grid = np.linspace(0, 1, 100).reshape((100, 1))
    y_grid = tanh_func(x_grid)['y']
    interp = LagrangeInterpolator(beta, x_var, model=tanh_func)
    interp.set_yi()
    y_interp = interp(x_grid)

    # Refine
    beta2 = (4,)
    interp2 = interp.refine(beta2)
    y2_interp = interp2(x_grid)

    # Compute errors
    N = 1000
    xtest = np.random.rand(N, 1)
    ytest = interp2(xtest)
    ytruth = tanh_func(xtest)['y']
    l2_error = np.sqrt(np.mean((ytest - ytruth) ** 2)) / np.sqrt(np.mean(ytruth ** 2))
    assert l2_error < 1e-1

    # Plot results
    if plots:
        fig, ax = plt.subplots()
        ax.plot(x_grid, y_grid, '-k', label='Model')
        ax.plot(interp.xi, interp.yi, 'or', markersize=6, label=r'Training data')
        ax.plot(x_grid, y_interp, '-r', label=r'$\beta=3$')
        ax.plot(interp2.xi, interp2.yi, 'ob', markersize=4)
        ax.plot(x_grid, y2_interp, '-b', label=r'$\beta=4$')
        ax_default(ax, r'Input', r'Output', legend=True)
        fig.tight_layout()
        plt.show()


def test_tensor_product_2d(plots=False):
    bb_2d_func = lambda x: nonlinear_wave(x, env_var=0.2**2, wave_amp=0.3)
    beta = (3, 3)
    x_vars = [UniformRV(0, 1), UniformRV(0, 1)]
    N = 50
    x_grid = np.linspace(0, 1, N)
    xg, yg = np.meshgrid(x_grid, x_grid)
    xg = xg.reshape((N, N, 1))
    yg = yg.reshape((N, N, 1))
    x = np.concatenate((xg, yg), axis=-1)
    z = bb_2d_func(x)['y']

    # Set up interpolator
    interp = LagrangeInterpolator(beta, x_vars, model=bb_2d_func)
    interp.set_yi()
    z_interp = interp(x)
    error = np.abs(z_interp - z)

    # Refine interpolator
    beta2 = (3, 4)
    interp2 = interp.refine(beta2)
    z2_interp = interp2(x)
    error2 = np.abs(z2_interp - z)
    vmin = min(np.min(z_interp), np.min(z), np.min(z2_interp))
    vmax = max(np.max(z_interp), np.max(z), np.max(z2_interp))
    emin = min(np.min(error), np.min(error2))
    emax = max(np.max(error), np.max(error2))

    # Compute errors
    N = 1000
    xtest = np.random.rand(N, 2)
    ytest = interp2(xtest)
    ytruth = bb_2d_func(xtest)['y']
    l2_error = np.sqrt(np.mean((ytest - ytruth)**2)) / np.sqrt(np.mean(ytruth**2))
    assert l2_error < 1e-1

    if plots:
        fig, ax = plt.subplots(2, 3)
        c1 = ax[0, 0].contourf(xg.squeeze(), yg.squeeze(), z.squeeze(), 60, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar(c1, ax=ax[0, 0])
        ax[0, 0].set_title('True function')
        ax_default(ax[0, 0], r'$x_1$', r'$x_2$', legend=False)
        c2 = ax[0, 1].contourf(xg.squeeze(), yg.squeeze(), z_interp.squeeze(), 60, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax[0, 1].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
        plt.colorbar(c2, ax=ax[0, 1])
        ax[0, 1].set_title('Interpolant')
        ax_default(ax[0, 1], r'$x_1$', '', legend=False)
        c3 = ax[0, 2].contourf(xg.squeeze(), yg.squeeze(), error.squeeze(), 60, cmap='viridis', vmin=emin, vmax=emax)
        ax[0, 2].plot(interp.xi[:, 0], interp.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
        plt.colorbar(c3, ax=ax[0, 2])
        ax[0, 2].set_title('Absolute error')
        ax_default(ax[0, 2], r'$x_1$', '', legend=False)
        c1 = ax[1, 0].contourf(xg.squeeze(), yg.squeeze(), z.squeeze(), 60, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar(c1, ax=ax[1, 0])
        ax[1, 0].set_title('True function')
        ax_default(ax[1, 0], r'$x_1$', r'$x_2$', legend=False)
        c2 = ax[1, 1].contourf(xg.squeeze(), yg.squeeze(), z2_interp.squeeze(), 60, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax[1, 1].plot(interp2.xi[:, 0], interp2.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
        plt.colorbar(c2, ax=ax[1, 1])
        ax[1, 1].set_title('Refined')
        ax_default(ax[1, 1], r'$x_1$', '', legend=False)
        c3 = ax[1, 2].contourf(xg.squeeze(), yg.squeeze(), error2.squeeze(), 60, cmap='viridis', vmin=emin, vmax=emax)
        ax[1, 2].plot(interp2.xi[:, 0], interp2.xi[:, 1], 'o', markersize=6, markerfacecolor='green')
        plt.colorbar(c3, ax=ax[1, 2])
        ax[1, 2].set_title('Absolute error')
        ax_default(ax[1, 2], r'$x_1$', '', legend=False)
        fig.set_size_inches(15, 11)
        fig.tight_layout()
        plt.show()
