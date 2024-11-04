"""Test interpolation classes. Currently, only barycentric Lagrange interpolation is supported."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from uqtils import approx_hess, approx_jac, ax_default

from amisc.examples.models import nonlinear_wave, tanh_func
from amisc.interpolator import Lagrange
from amisc.training import SparseGrid
from amisc.utils import relative_error
from amisc.variable import Variable


def test_tensor_product_1d(plots=False):
    """Test 1D tensor product Lagrange interpolation."""
    interp = Lagrange()
    x = Variable(distribution='U(0, 1)')
    alpha = ()          # no model fidelity for simple examples
    beta_interp = ()    # no beta for Lagrange interpolator
    domain = {'x': x.get_domain()}

    # Use sparse grid to get training data
    grid = SparseGrid()
    for beta in range(4):
        new_idx, new_x = grid.refine(alpha, (beta,), domain)
        new_y = tanh_func(new_x)
        grid.set(alpha, (beta,), new_idx, new_y)

    # Train the interpolator
    old_state = None
    training_data = grid.get(alpha, (3,))
    new_state = interp.refine(beta_interp, training_data, old_state, domain)

    x_grid = np.linspace(0, 1, 100)
    y_grid = tanh_func({'x': x_grid})['y']
    y_interp = interp.predict({'x': x_grid}, new_state, training_data)

    # Refine the sparse grid and predict again
    new_idx, new_x = grid.refine(alpha, (4,), domain)
    grid.set(alpha, (4,), new_idx, tanh_func(new_x))
    training_data = grid.get(alpha, (4,))
    new_state = interp.refine(beta_interp, training_data, new_state, domain)
    y2_interp = interp.predict({'x': x_grid}, new_state, training_data)

    # Compute errors
    N = 1000
    xtest = np.random.rand(N)
    ytest = interp.predict({'x': xtest}, new_state, training_data)['y']
    ytruth = tanh_func({'x': xtest})['y']
    l2_error = relative_error(ytest, ytruth)
    assert l2_error < 1e-1

    # Plot results
    if plots:
        fig, ax = plt.subplots()
        ax.plot(x_grid, y_grid, '-k', label='Model')
        xi, yi = grid.get(alpha, (3,))
        ax.plot(xi['x'], yi['y'], 'or', markersize=6, label=r'Training data')
        ax.plot(x_grid, y_interp['y'], '-r', label=r'$\beta=3$')
        xi, yi = grid.get(alpha, (4,))
        ax.plot(xi['x'], yi['y'], 'ob', markersize=4)
        ax.plot(x_grid, y2_interp['y'], '-b', label=r'$\beta=4$')
        ax_default(ax, r'Input', r'Output', legend=True)
        fig.tight_layout()
        plt.show()


def test_tensor_product_2d(plots=False):
    """Test 2D tensor product Lagrange interpolation."""
    bb_2d_func = lambda x: nonlinear_wave(x, env_var=0.2**2, wave_amp=0.3)
    interp = Lagrange()
    d = Variable(distribution='U(0, 1)')
    theta = Variable(distribution='U(0, 1)')
    alpha = ()          # no model fidelity for simple examples
    beta_interp = ()    # no beta for Lagrange interpolator
    domains = {'d': d.get_domain(), 'theta': theta.get_domain()}

    # Use sparse grid to get training data
    grid = SparseGrid()
    for beta in itertools.product(*[range(4) for _ in range(2)]):
        new_idx, new_x = grid.refine(alpha, beta, domains)
        new_y = bb_2d_func(new_x)
        grid.set(alpha, beta, new_idx, new_y)

    # Train the interpolator
    old_state = None
    training_data = grid.get(alpha, (3, 3))
    new_state = interp.refine(beta_interp, training_data, old_state, domains)

    # Predict
    N = 50
    x_grid = np.linspace(0, 1, N)
    xg, yg = np.meshgrid(x_grid, x_grid)
    zg = bb_2d_func({'d': xg, 'theta': yg})['y']
    z_interp = interp.predict({'d': xg, 'theta': yg}, new_state, training_data)['y']
    error = np.abs(z_interp - zg)

    # Refine interpolator
    new_beta = (3, 4)
    new_idx, new_x = grid.refine(alpha, new_beta, domains)
    grid.set(alpha, new_beta, new_idx, bb_2d_func(new_x))
    training_data = grid.get(alpha, new_beta)
    new_state = interp.refine(beta_interp, training_data, new_state, domains)
    z2_interp = interp.predict({'d': xg, 'theta': yg}, new_state, training_data)['y']
    error2 = np.abs(z2_interp - zg)

    vmin = min(np.min(z_interp), np.min(zg), np.min(z2_interp))
    vmax = max(np.max(z_interp), np.max(zg), np.max(z2_interp))
    emin = min(np.min(error), np.min(error2))
    emax = max(np.max(error), np.max(error2))

    # Compute errors
    N = 1000
    xtest = {'d': np.random.rand(N), 'theta': np.random.rand(N)}
    ytest = interp.predict(xtest, new_state, training_data)['y']
    ytruth = bb_2d_func(xtest)['y']
    l2_error = relative_error(ytest, ytruth)
    assert l2_error < 1e-1

    if plots:
        s = 100
        fig, ax = plt.subplots(2, 3)
        c1 = ax[0, 0].contourf(xg, yg, zg, 60, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar(c1, ax=ax[0, 0])
        ax[0, 0].set_title('True function')
        ax_default(ax[0, 0], r'$x_1$', r'$x_2$', legend=False)
        c2 = ax[0, 1].contourf(xg, yg, z_interp, 60, cmap='coolwarm', vmin=vmin,vmax=vmax)
        xi, yi = grid.get(alpha, (3, 3))
        ax[0, 1].scatter(xi['d'], xi['theta'], c=yi['y'], marker='o', s=s, cmap='coolwarm', vmin=vmin, vmax=vmax,
                         alpha=1, linewidths=2, edgecolors='black')
        plt.colorbar(c2, ax=ax[0, 1])
        ax[0, 1].set_title('Interpolant')
        ax_default(ax[0, 1], r'$x_1$', '', legend=False)
        c3 = ax[0, 2].contourf(xg, yg, error, 60, cmap='viridis', vmin=emin, vmax=emax)
        ax[0, 2].plot(xi['d'], xi['theta'], 'o', markersize=6, markerfacecolor='green')
        plt.colorbar(c3, ax=ax[0, 2])
        ax[0, 2].set_title('Absolute error')
        ax_default(ax[0, 2], r'$x_1$', '', legend=False)
        c1 = ax[1, 0].contourf(xg, yg, zg, 60, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar(c1, ax=ax[1, 0])
        ax[1, 0].set_title('True function')
        ax_default(ax[1, 0], r'$x_1$', r'$x_2$', legend=False)
        c2 = ax[1, 1].contourf(xg, yg, z2_interp, 60, cmap='coolwarm',vmin=vmin,vmax=vmax)
        xi, yi = grid.get(alpha, (3, 4))
        ax[1, 1].scatter(xi['d'], xi['theta'], c=yi['y'], marker='o', s=s, cmap='coolwarm', vmin=vmin, vmax=vmax,
                         alpha=1, linewidths=2, edgecolors='black')
        plt.colorbar(c2, ax=ax[1, 1])
        ax[1, 1].set_title('Refined')
        ax_default(ax[1, 1], r'$x_1$', '', legend=False)
        c3 = ax[1, 2].contourf(xg, yg, error2, 60, cmap='viridis', vmin=emin, vmax=emax)
        ax[1, 2].plot(xi['d'], xi['theta'], 'o', markersize=6, markerfacecolor='green')
        plt.colorbar(c3, ax=ax[1, 2])
        ax[1, 2].set_title('Absolute error')
        ax_default(ax[1, 2], r'$x_1$', '', legend=False)
        fig.set_size_inches(15, 11)
        fig.tight_layout()
        plt.show()


def test_interp_jacobian_and_hessian():
    f1 = lambda theta: 2 * theta['x1'] ** 2 + 3 * theta['x1'] * theta['x2'] ** 3 + np.cos(theta['x3'])
    f2 = lambda theta: (4 * theta['x1'] ** 2 + 2 * theta['x1'] ** 3 * theta['x2'] +
                        np.sin(theta['x3']) * theta['x1'])
    fun = lambda theta: {'y1': f1(theta), 'y2': f2(theta)}

    x1, x2, x3 = (Variable('x1', distribution='U(-2, 1)'), Variable('x2', distribution='U(-1, 2)'),
                  Variable('x3', distribution='U(-3.14, 3.14)'))
    interp = Lagrange()
    domains = {'x1': x1.get_domain(), 'x2': x2.get_domain(), 'x3': x3.get_domain()}
    alpha = ()
    beta_interp = ()
    beta_train = (3, 2, 4)

    # Use sparse grid to get training data
    grid = SparseGrid()
    for beta in itertools.product(*[range(b+1) for b in beta_train]):
        new_idx, new_x = grid.refine(alpha, beta, domains)
        new_y = fun(new_x)
        grid.set(alpha, beta, new_idx, new_y)

    old_state = None
    training_data = grid.get(alpha, beta_train)
    new_state = interp.refine(beta_interp, training_data, old_state, domains)

    N = (10, 11)

    def fun_vec(theta: np.ndarray):
        theta_dict = {var: theta[..., i] for i, var in enumerate(['x1', 'x2', 'x3'])}
        y_dict = fun(theta_dict)
        y_vec = np.concatenate([y_dict[var][..., np.newaxis] for var in ['y1', 'y2']], axis=-1)
        return y_vec

    xtest = {'x1': x1.sample(N), 'x2': x2.sample(N), 'x3': x3.sample(N)}
    xvec = np.concatenate([xtest[var][..., np.newaxis] for var in ['x1', 'x2', 'x3']], axis=-1)

    jac_truth = approx_jac(fun_vec, xvec)
    jac_interp = interp.gradient(xtest, new_state, training_data)
    jac_interp_vec = np.concatenate([np.expand_dims(jac_interp[var], axis=-2) for var in ['y1', 'y2']], axis=-2)
    assert np.allclose(jac_truth, jac_interp_vec, rtol=1e-2, atol=1e-2)

    hess_truth = approx_hess(fun_vec, xvec)
    hess_interp = interp.hessian(xtest, new_state, training_data)
    hess_interp_vec = np.concatenate([np.expand_dims(hess_interp[var], axis=-3) for var in ['y1', 'y2']], axis=-3)
    assert np.allclose(hess_truth, hess_interp_vec, rtol=1e-1, atol=1e-1)

    # Make sure interp gradient/hessian works directly at interpolation points as well
    xi, yi = training_data
    xtest = {'x1': x1.sample(10), 'x2': x2.sample(10), 'x3': x3.sample(10)}
    xi_test = {var: np.concatenate((xi[var], xtest[var]), axis=0) for var in ['x1', 'x2', 'x3']}
    N = xi_test['x1'].shape[0]
    idx = np.arange(0, N)
    np.random.shuffle(idx)
    for var in xi_test:
        xi_test[var] = xi_test[var][idx]
    xi_vec = np.concatenate([xi_test[var][..., np.newaxis] for var in ['x1', 'x2', 'x3']], axis=-1)

    jac_truth = approx_jac(fun_vec, xi_vec)
    jac_interp = interp.gradient(xi_test, new_state, training_data)
    jac_interp_vec = np.concatenate([np.expand_dims(jac_interp[var], axis=-2) for var in ['y1', 'y2']], axis=-2)
    assert np.allclose(jac_truth, jac_interp_vec, rtol=1e-2, atol=1e-2)

    hess_truth = approx_hess(fun_vec, xi_vec)
    hess_interp = interp.hessian(xi_test, new_state, training_data)
    hess_interp_vec = np.concatenate([np.expand_dims(hess_interp[var], axis=-3) for var in ['y1', 'y2']], axis=-3)
    assert np.allclose(hess_truth, hess_interp_vec, rtol=1e-1, atol=3e-1)
