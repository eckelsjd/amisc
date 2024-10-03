import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lapack
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
from uqtils import ax_default

# from amisc.examples.models import fire_sat_system
from amisc import Variable, Component, System
from amisc.utils import relative_error


# TODO: Include a swap and insert component test

def test_fire_sat(tmp_path, plots=False):
    """Test the fire satellite coupled system from Chaudhuri (2018)"""
    N = 100
    surr = fire_sat_system(save_dir=tmp_path)
    xt = surr.sample_inputs(N, use_pdf=True)
    yt = surr(xt, use_model='best')
    use_idx = ~np.any(np.isnan(yt), axis=-1)
    xt = xt[use_idx, :]
    yt = yt[use_idx, :]
    test_set = {'xt': xt, 'yt': yt}
    surr.fit(max_iter=5, max_tol=1e-2, max_runtime=1/12, test_set=test_set, num_refine=100)

    ysurr = surr.predict(xt)
    l2_error = np.sqrt(np.nanmean((ysurr-yt)**2, axis=0)) / np.sqrt(np.nanmean(yt**2, axis=0))
    assert np.nanmax(l2_error) < 0.2

    # Plot 1d slices
    slice_idx = ['H', 'Po', 'Cd']
    qoi_idx = ['Vsat', 'Asa', 'Pat']
    try:
        surr.plot_slice(slice_idx, qoi_idx, show_model=['best', 'worst'], model_dir=surr.root_dir,
                        random_walk=True, N=10)
    except np.linalg.LinAlgError:
        print('Its alright. Sometimes the random walks are wacky and FPI wont converge.')

    # Plot allocation bar charts
    surr.plot_allocation()

    if plots:
        # Plot error histograms
        fig, ax = plt.subplots(1, 3)
        ax[0].hist(yt[:, 0], color='red', bins=20, edgecolor='black', density=True, linewidth=1.2, label='Truth')
        ax[0].hist(ysurr[:, 0], color='blue', bins=20, edgecolor='black', density=True, linewidth=1.2, alpha=0.4,
                   label='Surrogate')
        ax[1].hist(yt[:, 7], color='red', bins=20, edgecolor='black', density=True, linewidth=1.2, label='Truth')
        ax[1].hist(ysurr[:, 7], color='blue', bins=20, edgecolor='black', density=True, linewidth=1.2, alpha=0.4,
                   label='Surrogate')
        ax[2].hist(yt[:, 8], color='red', bins=20, edgecolor='black', density=True, linewidth=1.2, label='Truth')
        ax[2].hist(ysurr[:, 8], color='blue', bins=20, edgecolor='black', density=True, linewidth=1.2, alpha=0.4,
                   label='Surrogate')
        ax_default(ax[0], 'Satellite velocity ($m/s$)', '', legend=True)
        ax_default(ax[1], 'Solar panel area ($m^2$)', '', legend=True)
        ax_default(ax[2], 'Attitude control power ($W$)', '', legend=True)
        fig.set_size_inches(9, 3)
        fig.tight_layout()

        plt.show()


def test_system_refine(plots=False):
    """Test iterative refinement for Figure 5 in Jakeman 2022"""
    def coupled_system():
        def f1(x):
            return {'y': x * np.sin(np.pi * x)}
        def f2(x):
            return {'y': 1 / (1 + 25 * x ** 2)}
        return f1, f2

    f1, f2 = coupled_system()
    exo_vars = [Variable(dist='U(0, 1)')]
    coupling_vars = [Variable(dist='U(0, 1)'), Variable(dist='U(0, 1)')]
    comp1 = ComponentSpec(f1, name='Model1', exo_in=0, coupling_out=0)
    comp2 = ComponentSpec(f2, name='Model2', coupling_in=0, coupling_out=1)
    surr = SystemSurrogate([comp1, comp2], exo_vars, coupling_vars, init_surr=True)

    Niter = 4
    x = np.linspace(0, 1, 100).reshape((100, 1))
    y1 = f1(x)['y']
    y2 = f2(x)['y']
    y3 = f2(y1)['y']
    fig, ax = plt.subplots(Niter, 3, sharex='col', sharey='row')
    for i in range(Niter):
        # Plot actual function values
        ax[i, 0].plot(x, y1, '-r', label='$f_1(x)$')
        ax[i, 1].plot(x, y2, '-r', label='$f_2(y)$')
        ax[i, 2].plot(x, y3, '-r', label='$f(x)$')

        # Plot first component surrogates
        comp = surr.get_component('Model1')
        ax[i, 0].plot(x, comp(x, training=True), '--k', label='$f_1$ current')
        beta_max = 0
        for alpha, beta in comp.index_set:
            if beta[0] > beta_max:
                beta_max = beta[0]
        interp = comp.get_sub_surrogate((), (beta_max,), include_grid=True)
        ax[i, 0].plot(interp.xi, interp.yi, 'ok', markersize=8, label='')
        for alpha, beta in comp.iterate_candidates():
            comp.update_misc_coeffs()
            yJ1 = surr(x, training=True)
            ax[i, 0].plot(x, comp(x, training=True), ':b', label='$f_1$ candidate')
            interp = comp.get_sub_surrogate(alpha, beta, include_grid=True)
            ax[i, 0].plot(interp.xi, interp.yi, 'xb', markersize=8, label='')
        comp.update_misc_coeffs()

        # Plot second component surrogates
        comp = surr.get_component('Model2')
        ax[i, 1].plot(x, comp(x, training=True), '--k', label='$f_2$ current')
        beta_max = 0
        for alpha, beta in comp.index_set:
            if beta[0] > beta_max:
                beta_max = beta[0]
        interp = comp.get_sub_surrogate((), (beta_max,), include_grid=True)
        ax[i, 1].plot(interp.xi, interp.yi, 'ok', markersize=8, label='')
        for alpha, beta in comp.iterate_candidates():
            comp.update_misc_coeffs()
            yJ2 = surr(x, training=True)
            ax[i, 1].plot(x, comp(x, training=True), '-.g', label='$f_2$ candidate')
            interp = comp.get_sub_surrogate(alpha, beta, include_grid=True)
            ax[i, 1].plot(interp.xi, interp.yi, 'xg', markersize=8, label='')
        comp.update_misc_coeffs()

        # Plot integrated surrogates
        ysurr = surr(x, training=True)
        ax[i, 2].plot(x, ysurr[:, 1:2], '--k', label='$f_J$')
        ax[i, 2].plot(x, yJ1[:, 1:2], ':b', label='$f_{J_1}$')
        ax[i, 2].plot(x, yJ2[:, 1:2], '-.g', label='$f_{J_2}$')
        ax_default(ax[i, 0], '$x$', '$f_1(x)$', legend=True)
        ax_default(ax[i, 1], '$y$', '$f_2(y)$', legend=True)
        ax_default(ax[i, 2], '$x$', '$f_2(f_1(x))$', legend=True)

        # Refine the system
        surr.refine(qoi_ind=None, num_refine=100)

    ysurr = surr.predict(x)
    ytrue = surr.predict(x, use_model='best')
    l2_error = np.sqrt(np.mean((ysurr-ytrue)**2, axis=0)) / np.sqrt(np.mean(ytrue**2, axis=0))
    assert np.max(l2_error) < 0.1

    if plots:
        fig.set_size_inches(3.5*3, 3.5*Niter)
        fig.tight_layout()
        plt.show()


def test_feedforward(plots=False):
    """Test MD system in Figure 5 in Jakeman 2022"""
    def f1(x):
        return {'y1': x['x'] * np.sin(np.pi * x['x'])}
    def f2(x):
        return {'y2': 1 / (1 + 25*x['y1']**2)}
    def f(x):
        return f2(f1(x))

    x = Variable(dist='U(0, 1)')
    y1 = Variable(dist='U(0, 1)')
    y2 = Variable(dist='U(0, 1)')
    comp1 = Component(f1, x, y1)
    comp2 = Component(f2, y1, y2)
    surr = System(comp1, comp2)

    x = {'x': np.linspace(0, 1, 100)}
    y1_ret = f1(x)
    y2_ret = f(x)
    y_truth = y2_ret.copy()
    y_truth.update(y1_ret)
    y_surr = surr(x, use_model='best')
    l2_error = np.array([relative_error(y_surr[var], y_truth[var]) for var in ['y1', 'y2']])
    assert np.all(l2_error < 1e-15)

    if plots:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout='tight')
        ax[0].plot(x['x'], y_truth['y1'], '-k', label='$f_1(x)$')
        ax[0].plot(x['x'], y_surr['y1'], '--r', label='Surrogate')
        ax_default(ax[0], '$x$', '$f(x)$', legend=True)
        ax[1].plot(x['x'], y_truth['y2'], '-k', label='$f(x)$')
        ax[1].plot(x['x'], y_surr['y2'], '--r', label='Surrogate')
        ax_default(ax[1], '$x$', '$f(x)$', legend=True)
        plt.show()


def test_system_surrogate(plots=False):
    """Test the MD system in figure 6 in Jakeman 2022"""
    def coupled_system(D1, D2, Q1, Q2):
        def f1(x, /, alpha=(0,)):
            eps = 10 ** (-float(alpha[0]))
            q = np.arange(1, Q1+1).reshape((1,)*len(x.shape[:-1]) + (Q1,))
            return {'y': (x[..., 0, np.newaxis] ** (q-1)) * np.sin(np.sum(x, axis=-1, keepdims=True) + eps)}

        def f2(x, /, alpha=(0,)):
            eps = 10 ** (-float(alpha[0]))
            q = np.arange(1, Q2+1).reshape((1,)*len(x.shape) + (Q2,))
            prod1 = np.prod(x[..., D2:, np.newaxis] ** (q) - eps, axis=-2)  # (..., Q2)
            prod2 = np.prod(x[..., :D2], axis=-1, keepdims=True)            # (..., 1)
            return {'y': prod1 * prod2}

        def f3(x, /, alpha=(0,), D3=D1):
            eps = 10 ** (-float(alpha[0]))
            prod1 = np.exp(-np.sum((x[..., D3:] - eps) ** 2, axis=-1))  # (...,)
            prod2 = 1 + (25/16)*np.sum(x[..., :D3], axis=-1) ** 2       # (...,)
            return {'y': np.expand_dims(prod1 / prod2, axis=-1)}        # (..., 1)

        def f(x):
            # Ground truth (arbitrary high alpha)
            alpha = (15,)
            x1 = x[..., :D1]
            y1 = f1(x1, alpha)['y']
            x2 = np.concatenate((x[..., D1:], y1), axis=-1)
            y2 = f2(x2, alpha)['y']
            x3 = np.concatenate((x1, y2), axis=-1)
            y3 = f3(x3, alpha)['y']
            return np.concatenate((y1, y2, y3), axis=-1)

        return f1, f2, f3, f

    # Hook up the 'wiring' for this example feedforward system
    D1 = 1
    D2 = D1
    Q1 = 1
    Q2 = Q1
    alpha = (15,)
    f1, f2, f3, f = coupled_system(D1, D2, Q1, Q2)
    comp1 = ComponentSpec(f1, name='Cathode', truth_alpha=alpha, exo_in=list(np.arange(0, D1)),
                          coupling_out=list(np.arange(0, Q1)), max_alpha=5, max_beta=(3,)*D1)
    comp2 = ComponentSpec(f2, name='Thruster', truth_alpha=alpha, exo_in=list(np.arange(D1, D1+D2)), max_alpha=5,
                          max_beta=(3,)*(D2+Q1), coupling_in={'Cathode': list(np.arange(0, Q1))},
                          coupling_out=list(np.arange(Q1, Q1+Q2)))
    comp3 = ComponentSpec(f3, name='Plume', truth_alpha=alpha, exo_in=list(np.arange(0, D1)), max_alpha=5,
                          coupling_in={'Thruster': list(np.arange(0, Q2))}, coupling_out=Q1+Q2, max_beta=(3,)*(D1+Q2))
    exo_vars = [Variable(dist='U(0, 1)') for i in range(D1+D2)]
    coupling_vars = [Variable(dist='U(0, 1)') for i in range(Q1+Q2+1)]
    surr = SystemSurrogate([comp1, comp2, comp3], exo_vars, coupling_vars, init_surr=False)

    # Test example
    N = 5000
    x = np.random.rand(N, D1+D2)
    y = f(x)
    y_surr = surr(x, use_model='best')
    l2_error = np.sqrt(np.mean((y_surr - y)**2, axis=0)) / np.sqrt(np.mean(y**2, axis=0))
    assert np.max(l2_error) < 1e-15

    # Show coupling variable pdfs
    if plots:
        fig, ax = plt.subplots()
        ls = ['-r', '--k', ':b']
        pts = np.linspace(0, 1, 100)
        for i in range(3):
            label_str = f'$\\rho(y_{{{i+1}}})$'
            kernel = gaussian_kde(y_surr[:, i])
            # ax[i].hist(y_surr[:, i], density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
            ax.plot(pts, kernel(pts), ls[i], label=label_str)
        ax_default(ax, r'$y$', 'PDF', legend=True)
        # fig.set_size_inches(9, 3)
        fig.tight_layout()
        plt.show()


def test_fpi():
    """Test fixed point iteration implementation against scipy fsolve."""
    f1 = lambda x: {'y1': -x['x']**3 + 2 * x['y2']**2}
    f2 = lambda x: {'y2': 3*x['x']**2 + 4 * x['y1']**(-2)}
    x = Variable('x', dist='U(0, 4)')
    y1 = Variable('y1', dist='U(1, 10)')
    y2 = Variable('y2', dist='U(1, 10)')
    comp1 = Component(f1, [x, y2], y1, name='m1')
    comp2 = Component(f2, [x, y1], y2, name='m2')
    surr = System(comp1, comp2)

    # Test on random x against scipy.fsolve
    N = 100
    tol = 1e-12
    x0 = np.array([5.5, 5.5])
    exo = surr.sample_inputs(N)
    y_surr = surr.predict(exo, use_model='best', anderson_mem=10, max_fpi_iter=200, fpi_tol=tol)  # (N, 2)
    nan_idx = np.full(N, False)
    for var, arr in y_surr.items():
        nan_idx = np.logical_or(nan_idx, np.isnan(arr))
    nan_idx = list(nan_idx.nonzero()[0])
    y_true = np.zeros((N, 2))
    bad_idx = []
    warnings.simplefilter('error')
    for i in range(N):
        def fun(x):
            y1 = x[0]
            y2 = x[1]
            res1 = -exo['x'][i]**3 + 2*y2**2 - y1
            res2 = 3*exo['x'][i]**2 + 4*y1**(-2) - y2
            return [res1, res2]

        try:
            y_true[i, :] = fsolve(fun, x0, xtol=tol)
        except Exception:
            bad_idx.append(i)

    y_surr = np.vstack([y_surr[var] for var in ['y1', 'y2']]).T
    y_surr = np.delete(y_surr, nan_idx + bad_idx, axis=0)
    y_true = np.delete(y_true, nan_idx + bad_idx, axis=0)
    l2_error = relative_error(y_surr, y_true)
    assert np.max(l2_error) < tol


def test_lls():
    """Test constrained linear least squares routine against scipy lapack."""
    X = 100
    Y = 100
    M = 10
    N = 10
    P = 1
    tol = 1e-8

    A = np.random.rand(X, Y, M, N)
    b = np.random.rand(X, Y, M, 1)
    C = np.random.rand(X, Y, P, N)
    d = np.random.rand(X, Y, P, 1)

    # custom solver
    t1 = time.time()
    alpha = np.squeeze(System._constrained_lls(A, b, C, d), axis=-1)  # (*, N)
    t2 = time.time()

    # Built in scipy solver
    alpha2 = np.zeros((X, Y, N))
    t3 = time.time()
    for i in range(X):
        for j in range(Y):
            Ai = A[i, j, ...]
            bi = b[i, j, ...]
            Ci = C[i, j, ...]
            di = d[i, j, ...]
            ret = lapack.dgglse(Ai, Ci, bi, di)
            alpha2[i, j, :] = ret[3]
    t4 = time.time()

    # Results
    diff = alpha - alpha2
    assert np.max(np.abs(diff)) < tol
    print(f'Custom CLLS time: {t2-t1} s. Scipy time: {t4-t3} s.')
