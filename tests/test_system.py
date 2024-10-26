"""Test `System` prediction and refinement methods."""
import copy
import time
import warnings
from concurrent.futures.process import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
from uqtils import ax_default

from amisc import Variable, Component, System
from amisc.utils import relative_error


def test_feedforward_simple(plots=False):
    """Test the MD system in Figure 5 in Jakeman 2022."""
    from amisc.examples.models import f1, f2

    surr = System(f1, f2)

    x = {'x': np.linspace(0, 1, 100)}
    y1_ret = f1(x['x'])
    y2_ret = f2(f1(x['x']))
    y_truth = {'y1': y1_ret, 'y2': y2_ret}
    y_surr = surr.predict(x, use_model='best')
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


def test_feedforward_three(plots=True):
    """Test the MD system in Figure 6 in Jakeman 2022."""
    def coupled_system(D1, D2, Q1, Q2):
        """Scalable number of inputs (D1, D2) and outputs (Q1, Q2)."""
        def f1(inputs, alpha=None):
            if alpha is None:
                alpha = np.ones((1, 1)) * 15
            eps = 10. ** (-alpha[..., 0])
            outputs = {}
            for q in range(1, Q1+1):
                sum = np.sum(np.concatenate([inputs[f'z1_{i}'][np.newaxis, ...] for i in range(1, D1+1)],
                                            axis=0), axis=0)
                outputs[f'y1_{q}'] = inputs['z1_1'] ** q * np.sin(sum + eps)
            return outputs

        def f2(inputs, alpha=None):
            if alpha is None:
                alpha = np.ones((1, 1)) * 15
            eps = 10. ** (-alpha[..., 0])
            outputs = {}
            for q in range(1, Q2+1):
                prod1 = np.prod(np.concatenate([inputs[f'y1_{i}'][np.newaxis, ...] ** (q+1) - eps
                                                for i in range(1, Q1+1)], axis=0), axis=0)
                prod2 = np.prod(np.concatenate([inputs[f'z2_{i}'][np.newaxis, ...] for i in range(1, D2+1)],
                                               axis=0), axis=0)
                outputs[f'y2_{q}'] = prod1 * prod2
            return outputs

        def f3(inputs, alpha=None, D3=D1):
            if alpha is None:
                alpha = np.ones((1, 1)) * 15
            eps = 10. ** (-alpha[..., 0])
            sum1 = np.sum(np.concatenate([(inputs[f'y2_{i}'][np.newaxis, ...] - eps) ** 2 for i in range(1, Q2+1)],
                                         axis=0), axis=0)
            sum2 = np.sum(np.concatenate([inputs[f'z1_{i}'][np.newaxis, ...] for i in range(1, D3+1)], axis=0), axis=0)
            y3 = np.exp(-sum1) / (1 + (25/16) * sum2 ** 2)
            return {'y3_1': y3}

        return f1, f2, f3

    # Hook up the 'wiring' for this example feedforward system
    D1 = 1
    D2 = D1
    Q1 = 1
    Q2 = Q1
    alpha = (15,)
    inputs = ([Variable(f'z1_{i}', dist='U(0, 1)') for i in range(1, D1+1)] +
              [Variable(f'z2_{i}', dist='U(0, 1)') for i in range(1, D2+1)])
    outputs = ([Variable(f'y1_{i}') for i in range(1, Q1+1)] + [Variable(f'y2_{i}') for i in range(1, Q2+1)] +
               [Variable('y3_1')])
    f1, f2, f3 = coupled_system(D1, D2, Q1, Q2)
    comp1 = Component(f1, inputs[:D1], outputs[:Q1], vectorized=True, max_alpha=alpha)
    comp2 = Component(f2, inputs[D1:] + outputs[:Q1], outputs[Q1:Q1+Q2], vectorized=True, max_alpha=alpha)
    comp3 = Component(f3, inputs[:D1] + outputs[Q1:Q1+Q2], outputs[Q1+Q2:], vectorized=True, max_alpha=alpha)
    surr = System(comp1, comp2, comp3)

    # Test example
    N = 5000
    z = surr.sample_inputs(N)
    all_inputs = copy.deepcopy(z)
    y1_truth = f1(z)
    all_inputs.update(y1_truth)
    y2_truth = f2(all_inputs)
    all_inputs.update(y2_truth)
    y3_truth = f3(all_inputs)
    y1_truth.update({**y2_truth, **y3_truth})
    y_surr = surr.predict(z, use_model='best')

    for var in y1_truth:
        l2_error = relative_error(y_surr[var], y1_truth[var])
        assert np.max(l2_error) < 1e-15

    # Show coupling variable pdfs
    if plots:
        # Typo in the figure: rho(y1) is flipped and rho(y3) is not so exaggerated
        fig, ax = plt.subplots()
        ls = ['-r', '--k', ':b']
        pts = np.linspace(0, 1, 100)
        for i, var in enumerate(y1_truth):
            label_str = f'$\\rho(y_{{{i+1}}})$'
            kernel = gaussian_kde(y_surr[var])
            # ax[i].hist(y_surr[:, i], density=True, bins=20, color='r', edgecolor='black', linewidth=1.2)
            ax.plot(pts, kernel(pts), ls[i], label=label_str)
        ax.set_xlabel(r'$y$')
        ax.set_ylabel('PDF')
        ax.legend()
        fig.tight_layout()
        plt.show()


def test_field_quantity():
    # TODO
    pass


def test_fpi():
    """Test fixed point iteration implementation against scipy fsolve."""
    f1 = lambda x: {'y1': -x['x']**3 + 2 * x['y2']**2}
    f2 = lambda x: {'y2': 3*x['x']**2 + 4 * x['y1']**(-2)}
    x = Variable('x', dist='U(1, 4)')
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


def test_fpi_field_quantity():
    # TODO
    pass


def test_system_refine(plots=False):
    """Test iterative refinement for Figure 5 in Jakeman 2022."""
    from amisc.examples.models import f1, f2

    beta_max = (4,)
    surr = System(Component(f1, name='f1', vectorized=True, max_beta_train=beta_max),
                  Component(f2, name='f2', vectorized=True, max_beta_train=beta_max))

    for var in surr.variables():
        var.domain = (0, 1)

    Niter = 4
    x = np.linspace(0, 1, 100)
    y1 = f1(x)
    y2 = f2(x)
    y3 = f2(y1)

    # Initialize all components
    for i in range(len(surr.components)):
        surr.refine()

    fig, ax = plt.subplots(Niter, 3, sharex='col', sharey='row')
    for i in range(Niter):
        # Plot actual function values
        ax[i, 0].plot(x, y1, '-r', label='$f_1(x)$')
        ax[i, 1].plot(x, y2, '-r', label='$f_2(y)$')
        ax[i, 2].plot(x, y3, '-r', label='$f(x)$')

        # Plot first component surrogates
        comp = surr.get_component('f1')
        ax[i, 0].plot(x, comp.predict({'x': x}, index_set='train')['y1'], '--k', label='$f_1$ current')
        beta_max = 0
        for alpha, beta in comp.active_set:
            if beta[0] > beta_max:
                beta_max = beta[0]
        xi, yi = comp.get_training_data((), (beta_max,))
        ax[i, 0].plot(xi['x'], yi['y1'], 'ok', markersize=8, label='')
        for alpha, beta in comp.candidate_set:
            yJ1 = surr.predict({'x': x}, index_set={comp.name: {(alpha, beta)}}, incremental={comp.name: True})['y2']
            y1_cand = comp.predict({'x': x}, index_set={(alpha, beta)}, incremental=True)['y1']
            ax[i, 0].plot(x, y1_cand, ':b', label='$f_1$ candidate')
            xi, yi = comp.get_training_data(alpha, beta)
            ax[i, 0].plot(xi['x'], yi['y1'], 'xb', markersize=8, label='')

        # Plot second component surrogates
        comp = surr.get_component('f2')
        ax[i, 1].plot(x, comp.predict({'y1': x}, index_set='train')['y2'], '--k', label='$f_2$ current')
        beta_max = 0
        for alpha, beta in comp.active_set:
            if beta[0] > beta_max:
                beta_max = beta[0]
        xi, yi = comp.get_training_data((), (beta_max,))
        ax[i, 1].plot(xi['y1'], yi['y2'], 'ok', markersize=8, label='')
        for alpha, beta in comp.candidate_set:
            yJ2 = surr.predict({'x': x}, index_set={comp.name: {(alpha, beta)}}, incremental={comp.name: True})['y2']
            y2_cand = comp.predict({'y1': x}, index_set={(alpha, beta)}, incremental=True)['y2']
            ax[i, 1].plot(x, y2_cand, '-.g', label='$f_2$ candidate')
            xi, yi = comp.get_training_data(alpha, beta)
            ax[i, 1].plot(xi['y1'], yi['y2'], 'xg', markersize=8, label='')

        # Plot integrated surrogates
        ysurr = surr.predict({'x': x}, index_set='train')['y2']
        ax[i, 2].plot(x, ysurr, '--k', label='$f_J$')
        ax[i, 2].plot(x, yJ1, ':b', label='$f_{J_1}$')
        ax[i, 2].plot(x, yJ2, '-.g', label='$f_{J_2}$')
        ax_default(ax[i, 0], '$x$', '$f_1(x)$', legend=True)
        ax_default(ax[i, 1], '$y$', '$f_2(y)$', legend=True)
        ax_default(ax[i, 2], '$x$', '$f_2(f_1(x))$', legend=True)

        # Refine the system
        surr.refine()

    ysurr = surr.predict({'x': x})
    ytrue = surr.predict({'x': x}, use_model='best')
    for var in ysurr:
        l2_error = relative_error(ysurr[var], ytrue[var])
        assert np.max(l2_error) < 0.1

    if plots:
        fig.set_size_inches(3.5*3, 3.5*Niter)
        fig.tight_layout()
        plt.show()


def test_fire_sat(tmp_path, plots=True):
    """Test the fire satellite coupled system from Chaudhuri (2018)."""
    from amisc.examples.models import fire_sat_system

    N = 100
    surr = fire_sat_system(save_dir=tmp_path)

    targs = ['Slew', 'Asa', 'Pat']
    xt = surr.sample_inputs(N, use_pdf=True)
    yt = surr.predict(xt, use_model='best', targets=targs)
    use_idx = np.full(N, True)
    for var, arr in yt.items():
        use_idx = np.logical_and(use_idx, ~np.isnan(arr))

    xt = {var: arr[use_idx] for var, arr in xt.items()}
    yt = {var: arr[use_idx] for var, arr in yt.items()}

    # with ProcessPoolExecutor(max_workers=4) as executor:
    executor=None
    surr.fit(targets=targs, max_iter=10, max_tol=1e-3, runtime_hr=4/12, test_set=(xt, yt), executor=executor,
             plot_interval=4, estimate_bounds=True)

    ysurr = surr.predict(xt, targets=targs)
    for var in yt:
        l2_error = relative_error(ysurr[var], yt[var])
        assert np.nanmax(l2_error) < 0.2

    # Plot 1d slices
    slice_idx = ['H', 'Po', 'Cd']
    try:
        surr.plot_slice(slice_idx, targs, show_model=['best', 'worst'], model_dir=surr.root_dir,
                        random_walk=True, num_steps=15)
    except np.linalg.LinAlgError:
        print('Its alright. Sometimes the random walks are wacky and FPI wont converge.')

    # Plot allocation bar charts
    # surr.plot_allocation()

    if plots:
        # Plot error histograms
        fig, ax = plt.subplots(1, 3)
        kwargs = {'bins': 20, 'edgecolor': 'black', 'density': False, 'linewidth': 1.2}
        for i, var in enumerate(targs):
            ax[i].hist(yt[var], color='r', label='Truth', **kwargs)
            ax[i].hist(ysurr[var], color='b', alpha=0.4, label='Surrogate', **kwargs)
            ax_default(ax[i], surr.outputs()[var].get_tex(symbol=False, units=True), '', legend=True)

        fig.set_size_inches(9, 3)
        fig.tight_layout()
        plt.show()
