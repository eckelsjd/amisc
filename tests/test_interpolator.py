"""Test interpolation classes. Currently, only Lagrange interpolation and Linear regression are supported."""
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PolynomialFeatures
from uqtils import approx_hess, approx_jac, ax_default

from amisc.examples.models import nonlinear_wave, tanh_func
from amisc.interpolator import GPR, Lagrange, Linear
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


def test_sklearn_linear():
    """Test linear regression classes with sklearn."""
    # Test simple ridge (no noise)
    num_inputs = 4
    num_outputs = 3
    num_train = 50
    num_test = 20
    noise_std = 0.0
    true_coeff = np.random.rand(num_inputs + 1, num_outputs) * 2 - 1  # includes intercept term (num_inputs + 1)

    def linear_model(inputs, noise_std=0.0):
        """Compute a linear model."""
        x_mat = np.concatenate([inputs[var][..., np.newaxis] for var in inputs], axis=-1)
        y_mat = np.dot(x_mat, true_coeff[:-1, :]) + true_coeff[-1, :]
        if noise_std >= 0:
            y_mat += np.random.randn(*y_mat.shape) * noise_std

        return {f'y{i}': y_mat[..., i] for i in range(num_outputs)}

    xtrain = {f'x{i}': np.random.rand(num_train) * 2 - 1 for i in range(num_inputs)}
    ytrain = linear_model(xtrain, noise_std=noise_std)

    interp = Linear(regressor='Ridge', regressor_opts={'alpha': 0.0})
    state = interp.refine((), (xtrain, ytrain), None, {})

    # 1d plot for quick comparisons (only for xdim=1, ydim=2)
    # fig, ax = plt.subplots()
    # xlin = np.linspace(-1, 1, 100)
    # ypred = interp.predict({'x0': xlin}, state, ())
    # ax.scatter(xtrain['x0'], ytrain['y0'], color='k')
    # ax.scatter(xtrain['x0'], ytrain['y1'], color='r')
    # ax.plot(xlin, true_coeff[0, 0] * xlin + true_coeff[1, 0], '-k')
    # ax.plot(xlin, ypred['y0'], '--b')
    # ax.plot(xlin, true_coeff[0, 1] * xlin + true_coeff[1, 1], '-r')
    # ax.plot(xlin, ypred['y1'], '--g')
    # plt.show()

    xtest = {f'x{i}': np.random.rand(num_test) for i in range(num_inputs)}
    ytest = linear_model(xtest, noise_std=noise_std)
    ypred = interp.predict(xtest, state, ())

    # Should exactly fit linear model with no noise
    assert relative_error(state.regressor['linear'].coef_.T, true_coeff[:-1, :]) < 1e-8
    assert relative_error(state.regressor['linear'].intercept_, true_coeff[-1, :]) < 1e-8
    assert all([relative_error(ypred[var], ytest[var]) < 1e-8 for var in ypred])

    # Test other linear regressions with noise
    regressors = {'Lasso': {'alpha': 0.001},
                  'ElasticNet': {'alpha': 0.001, 'l1_ratio': 0.3},
                  'Lars': {'eps': 1e-10}}
    noise_std = 0.02
    ytrain = linear_model(xtrain, noise_std=noise_std)

    for regressor, opts in regressors.items():
        interp = Linear(regressor=regressor, regressor_opts=opts)
        state = interp.refine((), (xtrain, ytrain), None, {})

        ypred = interp.predict(xtest, state, ())
        err = {var: relative_error(ypred[var], ytest[var]) for var in ypred}
        assert all([err[var] < 0.05 for var in err])


def test_sklearn_polynomial():
    """Test sklearn polynomial regression (and also cross-validation for hyperparameter tuning)."""
    rng = np.random.default_rng(42)
    num_train = 100
    num_test = (20, 2)
    num_inputs = 3
    polynomial_opts = {'degree': 3, 'include_bias': False}

    # Generate random polynomial coefficients
    feat = PolynomialFeatures(**polynomial_opts)
    feat.fit(rng.random((num_train, num_inputs)))
    powers = feat.powers_  # (num_features, num_inputs) -- gives input powers for each polynomial feature
    true_coeff = rng.random(powers.shape[0]) * 2 - 1
    zero_ind = rng.integers(0, powers.shape[0], size=powers.shape[0] // 2)  # zero half the coefficients
    true_coeff[zero_ind] = 0
    true_intercept = rng.random() * 2 - 1

    def polynomial_model(inputs, noise_std=0.0):
        """Compute a linear model with polynomial features."""
        x_mat = np.concatenate([inputs[var][..., np.newaxis] for var in inputs], axis=-1)
        y_mat = np.zeros(x_mat.shape[:-1])

        for i, feature in enumerate(powers):
            monomial = np.ones(x_mat.shape[:-1]) * true_coeff[i]
            for j, power in enumerate(feature):
                monomial *= x_mat[..., j] ** power

            y_mat += monomial

        y_mat += true_intercept

        if noise_std >= 0:
            y_mat += np.random.randn(*y_mat.shape) * noise_std

        return {'y': y_mat}

    xtrain = {f'x{i}': rng.random(num_train) * 2 - 1 for i in range(num_inputs)}
    ytrain = polynomial_model(xtrain, noise_std=0.02)

    interp = Linear(regressor='RidgeCV', regressor_opts={'alphas': np.logspace(-4, 2, 7)},
                    polynomial_opts=polynomial_opts)
    state = interp.refine((), (xtrain, ytrain), None, {})

    xtest = {f'x{i}': rng.random(num_test) * 2 - 1 for i in range(num_inputs)}
    ytest = polynomial_model(xtest, noise_std=0.0)
    ypred = interp.predict(xtest, state, ())

    coeff_err = relative_error(state.regressor['linear'].coef_, true_coeff)
    intercept_err = relative_error(state.regressor['linear'].intercept_, true_intercept)
    err = relative_error(ypred['y'], ytest['y'])

    tol = 0.09
    assert coeff_err < tol, f'Error in polynomial coefficients: {coeff_err} > {tol}'
    assert intercept_err < tol, f'Error in polynomial intercept: {intercept_err} > {tol}'
    assert err < tol, f'Error in polynomial prediction: {err} > {tol}'


def test_GPR_1d(plots=False):
    """Test GPR interpolation for no noise 1D function."""
    num_train = 30
    num_test = 40

    def model(inputs):
        x1 = inputs['x1']
        y = (6 * x1 - 2) ** 2 * np.sin(12 * x1 - 4)
        return {'y': y}

    xtrain = {'x1': np.linspace(0, 1, num_train)}
    ytrain = model(xtrain)

    interp = GPR(kernel='RBF', kernel_opts={'length_scale': 0.1, 'length_scale_bounds': 'fixed'})
    state = interp.refine((), (xtrain, ytrain), None, {})

    xtest = {'x1': np.linspace(0, 1, num_test)}
    ytest = model(xtest)
    ypred = interp.predict(xtest, state, ())

    l2_error = {var: relative_error(ypred[var], ytest[var]) for var in ypred}
    assert all([l2_error[var] < 0.01 for var in l2_error]), f'L2 error {l2_error} is greater than 0.01'

    if plots:
        plt.plot(xtest['x1'], ytest['y'], 'r--', label='Model')
        plt.plot(xtest['x1'], ypred['y'], 'k', label='Surrogate')
        plt.scatter(xtrain['x1'], ytrain['y'], marker='o', s=25, color='blue', label='Training data')
        plt.legend()
        plt.show()

    # Test for a few other kernels
    regressors = {'Matern': {'length_scale': 0.1, 'nu': 2.5},
                  'RationalQuadratic': {'length_scale': 1, 'alpha': 1, 'alpha_bounds': 'fixed'},
                  'ExpSineSquared': {'length_scale': 1e-3, 'length_scale_bounds': 'fixed',
                                         'periodicity': 1000, 'periodicity_bounds': 'fixed'}}
    for regressor, opts in regressors.items():
        interp = GPR(kernel=regressor, kernel_opts=opts)
        state = interp.refine((), (xtrain, ytrain), None, {})

        ypred = interp.predict(xtest, state, ())
        err = {var: relative_error(ypred[var], ytest[var]) for var in ypred}
        assert all([err[var] < 0.01 for var in err]), f'L2 error {err} is greater than 0.01 for {regressor}'


def test_GPR_2d(plots=False):
    """Test GPR interpolator with 2D Branin Function (https://www.sfu.ca/~ssurjano/branin.html)"""
    rng = np.random.default_rng(41)
    num_train = 150
    num_test = 30
    noise_std = 8

    def model(inputs, noise_std):
        """Branin function."""
        x1 = inputs['x1']
        x2 = inputs['x2']
        input_shape = np.atleast_1d(x1).shape
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        y = (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s +
             noise_std * np.random.randn(*input_shape))
        return {'y': y}

    xtrain = {'x1': rng.uniform(-5, 10, num_train), 'x2': rng.uniform(0, 15, num_train)}
    ytrain = model(xtrain, noise_std)
    interp = GPR(kernel=['Sum',
                         ['Product',
                          ['ConstantKernel', {'constant_value': 100, 'constant_value_bounds': (1000, 100000)}],
                          ['RBF', {'length_scale': 5, 'length_scale_bounds': (1, 10)}]],
                         ['WhiteKernel', {'noise_level': noise_std, 'noise_level_bounds': (0.1 * noise_std,
                                                                                           10 * noise_std)}]])

    with warnings.catch_warnings(action='ignore', category=ConvergenceWarning):
        state = interp.refine((), (xtrain, ytrain), None, {})

    xtest = {'x1': rng.uniform(-5, 10, num_test), 'x2': rng.uniform(0, 15, num_test)}
    ytest = model(xtest, noise_std=0)

    ypred = interp.predict(xtest, state, ())

    l2_error = {var: relative_error(ypred[var], ytest[var]) for var in ypred}
    rel_noise = 2 * noise_std / np.percentile(ytrain['y'], 75)  # pretty skewed so use percentile
    assert l2_error['y'] < rel_noise, f'L2 error: {l2_error["y"]} is greater than relative noise {rel_noise}'

    if plots:
        test_x1 = np.linspace(-5, 10, num_test)
        test_x2 = np.linspace(0, 15, num_test)
        X1, X2 = np.meshgrid(test_x1, test_x2)
        true_Y = model({'x1': X1, 'x2': X2}, noise_std=0)['y']
        pred_Y = interp.predict({'x1': X1, 'x2': X2}, state, ())['y']

        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=min(true_Y.min(), pred_Y.min()), vmax=max(true_Y.max(), pred_Y.max()))
        fig, axs = plt.subplots(1, 2, figsize=(11, 5), layout='tight')
        axs[0].contourf(X1, X2, true_Y, levels=10, cmap='viridis', norm=norm)
        axs[0].set_title('True Function')
        axs[0].set_xlabel('x1')
        axs[0].set_ylabel('x2', rotation=0)
        axs[0].set_xlim(-5.1, 10.1)
        axs[0].set_ylim(-0.1, 15.1)
        contour2 = axs[1].contourf(X1, X2, pred_Y, levels=10, cmap='viridis', norm=norm)
        axs[1].scatter(xtrain['x1'], xtrain['x2'], c=ytrain['y'], marker='o', s=25, cmap='viridis',
                       norm=norm, alpha=1, linewidths=2, edgecolors='black')
        axs[1].set_xlabel('x1')
        axs[1].set_ylabel('x2', rotation=0)
        axs[1].set_xlim(-5.1, 10.1)
        axs[1].set_ylim(-0.1, 15.1)
        axs[1].set_title('GPR surrogate')
        cbar = fig.colorbar(contour2, ax=axs[1], orientation='vertical')
        cbar.set_label('y', rotation=0)
        plt.show()


def test_GPR_nd():
    """Multi-dimensionsal GPR test with noise"""
    num_train = 150
    num_test = 30
    noise_std = 0.1

    def model(inputs, noise_std=0.0):
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']

        y1 = x1 + 0.01 * x2 + 0.1 * x3 + np.random.randn(*x1.shape) * noise_std
        y2 = np.sin(np.pi * x1) + 0.001 * x2 + 0.5 * x4 + np.random.randn(*x1.shape) * noise_std
        y3 = x1 * x4 + 0.05 * x3**2 + np.random.randn(*x1.shape) * noise_std

        return {'y1': y1, 'y2': y2, 'y3': y3}

    xtrain = {'x1': np.random.uniform(0, 1, num_train),
              'x2': np.random.uniform(100, 500, num_train),
              'x3': np.random.uniform(-10, 10, num_train),
              'x4': np.random.uniform(0.01, 0.1, num_train)}
    ytrain = model(xtrain, noise_std=noise_std)

    interp = GPR(kernel=['Sum', ['RBF', {'length_scale': [0.1, 10, 1, 0.1]}], ['WhiteKernel']])
    state = interp.refine((), (xtrain, ytrain), None, {})
    xtest = {'x1': np.random.uniform(0, 1, num_test),
             'x2': np.random.uniform(100, 500, num_test),
             'x3': np.random.uniform(-10, 10, num_test),
             'x4': np.random.uniform(0.01, 0.1, num_test)}
    ytest = model(xtest, noise_std=0)
    ypred = interp.predict(xtest, state, ())
    l2_error = {var: relative_error(ypred[var], ytest[var]) for var in ypred}
    rel_noise = {var: np.abs(2 * noise_std / np.mean(ytest[var])) for var in ytest}
    assert all([l2_error[var] < rel_noise[var] for var in l2_error]), f'L2 error: {l2_error} is \
        greater than  relative noise {rel_noise}'
