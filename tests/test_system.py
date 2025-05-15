"""Test `System` prediction and refinement methods."""
import copy
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
from sklearn.exceptions import ConvergenceWarning
from uqtils import ax_default

from amisc import Component, System, Variable
from amisc.compression import SVD
from amisc.interpolator import Interpolator
from amisc.utils import relative_error


def test_sample_inputs():
    def my_func(inputs):
        return inputs

    inputs = [Variable('x1', distribution='N(0, 1)', category='calibration'),
              Variable('x2', domain=(0.1, 100), norm='log10', category='calibration'),
              Variable('x3', domain=(1e-6, 10e-6), norm='linear(1e6)', category='other')]
    surr = System(Component(my_func, inputs, ['y1', 'y2', 'y3'], name='c1'))

    s1 = surr.sample_inputs(100)
    assert all([v in s1 for v in ['x1', 'x2', 'x3']])
    assert np.all(s1['x1'] < 3) and np.all(s1['x1'] > -3)
    assert np.all(s1['x2'] < 2) and np.all(s1['x2'] > -1)
    assert np.all(s1['x3'] < 10) and np.all(s1['x3'] > 1)

    s1 = surr.sample_inputs(100, use_pdf='x1', include='calibration', normalize=False)
    assert 'x3' not in s1
    assert np.all(s1['x2'] < 100) and np.all(s1['x2'] > 0.1)

    s1 = surr.sample_inputs(100, exclude=['x1', 'x2'], normalize=False)
    assert 'x1' not in s1 and 'x2' not in s1
    assert np.all(s1['x3'] < 10e-6) and np.all(s1['x3'] > 1e-6)


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


def test_feedforward_three(plots=False):
    """Test the MD system in Figure 6 in Jakeman 2022."""
    def coupled_system(D1, D2, Q1, Q2):
        """Scalable number of inputs (D1, D2) and outputs (Q1, Q2)."""
        def f1(inputs, model_fidelity=None):
            if model_fidelity is None:
                model_fidelity = np.ones((1, 1)) * 15
            eps = 10. ** (-model_fidelity[..., 0])
            outputs = {}
            for q in range(1, Q1+1):
                sum = np.sum(np.concatenate([inputs[f'z1_{i}'][np.newaxis, ...] for i in range(1, D1+1)],
                                            axis=0), axis=0)
                outputs[f'y1_{q}'] = inputs['z1_1'] ** q * np.sin(sum + eps)
            return outputs

        def f2(inputs, model_fidelity=None):
            alpha = model_fidelity
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

        def f3(inputs, model_fidelity=None, D3=D1):
            alpha = model_fidelity
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
    inputs = ([Variable(f'z1_{i}', distribution='U(0, 1)') for i in range(1, D1+1)] +
              [Variable(f'z2_{i}', distribution='U(0, 1)') for i in range(1, D2+1)])
    outputs = ([Variable(f'y1_{i}') for i in range(1, Q1+1)] + [Variable(f'y2_{i}') for i in range(1, Q2+1)] +
               [Variable('y3_1')])
    f1, f2, f3 = coupled_system(D1, D2, Q1, Q2)
    comp1 = Component(f1, inputs[:D1], outputs[:Q1], vectorized=True, model_fidelity=alpha)
    comp2 = Component(f2, inputs[D1:] + outputs[:Q1], outputs[Q1:Q1+Q2], vectorized=True, model_fidelity=alpha)
    comp3 = Component(f3, inputs[:D1] + outputs[Q1:Q1+Q2], outputs[Q1+Q2:], vectorized=True, model_fidelity=alpha)
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


def test_fpi():
    """Test fixed point iteration implementation against scipy fsolve."""
    f1 = lambda x: {'y1': -x['x']**3 + 2 * x['y2']**2}
    f2 = lambda x: {'y2': 3*x['x']**2 + 4 * x['y1']**(-2)}
    x = Variable('x', distribution='U(1, 4)')
    y1 = Variable('y1', distribution='U(1, 10)')
    y2 = Variable('y2', distribution='U(1, 10)')
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


def test_system_refine(plots=False):
    """Test iterative refinement for Figure 5 in Jakeman 2022."""
    from amisc.examples.models import f1, f2

    beta_max = (4,)
    surr = System(Component(f1, name='f1', vectorized=True, data_fidelity=beta_max),
                  Component(f2, name='f2', vectorized=True, data_fidelity=beta_max))

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


def test_simulate_fit():
    """Test looping back through training history and recomputing on a test set."""
    from amisc.examples.models import f1, f2

    beta_max = (4,)
    surr = System(Component(f1, name='f1', vectorized=True, data_fidelity=beta_max),
                  Component(f2, name='f2', vectorized=True, data_fidelity=beta_max))

    for var in surr.variables():
        var.domain = (0, 1)

    xtest = {'x': np.linspace(0, 1, 100)}
    y1_ret = f1(xtest['x'])
    y2_ret = f2(f1(xtest['x']))
    ytest = {'y1': y1_ret, 'y2': y2_ret}

    max_iter = 8
    surr.fit(max_iter=max_iter, test_set=(xtest, ytest))

    i = 0
    for train_res, active_set, candidate_set, misc_coeff_train, misc_coeff_test in surr.simulate_fit():
        error = train_res['test_error']
        if all([~np.isnan(val) for val in error.values()]):
            test_index_set = {comp: active_set[comp].union(candidate_set[comp]) for comp in active_set}
            y_surr_train = surr.predict(xtest, index_set=active_set, misc_coeff=misc_coeff_train)
            y_surr_test = surr.predict(xtest, index_set=test_index_set, misc_coeff=misc_coeff_test)

            train_error = {var: relative_error(y_surr_train[var], ytest[var]) for var in ytest}
            test_error = {var: relative_error(y_surr_test[var], ytest[var]) for var in ytest}

            # Testing error should be the same as was originally computed during fit
            assert all([np.allclose(error[var], test_error[var]) for var in train_error])

            # Error using the candidate set should be less than the training error
            assert all([np.all(test_error[var] <= train_error[var]) for var in test_error])

        i += 1

    assert i == max_iter


def test_fit_with_executor(model_cost=1, max_iter=5, max_workers=8):
    """Run fit with an executor."""

    def expensive_model(inputs, model_cost=1):
        time.sleep(model_cost)
        return {'y': -inputs['x1'] ** 3 + 2 * inputs['x2'] ** 2 + inputs['x3']}

    inputs = [Variable('x1', distribution='U(0, 1)'), Variable('x2', distribution='U(0, 1)'),
              Variable('x3', distribution='U(0, 1)')]
    outputs = Variable('y')
    comp = Component(expensive_model, inputs, outputs, data_fidelity=(3, 3, 3), model_cost=model_cost)
    surr = System(comp)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        surr.fit(max_iter=max_iter, executor=executor)


def test_predict_extra_returns():
    """Test system prediction when component models return extra things."""
    class ExtraThing:
        def __init__(self, x):
            self.x = x

    def comp1(inputs):
        return {'y1': -inputs['x']**2, 'extra': inputs['x'], 'extrathing': ExtraThing(inputs['x'])}
    def comp2(inputs):
        return {'y2': 2*inputs['y1'], 'more_extra': np.random.rand(100, 2)}

    surr = System(Component(comp1, [Variable('x', domain=(0, 1))], [Variable('y1', domain=(-1, 0))], data_fidelity=(3,),
                            name='comp1'),
                  Component(comp2, ['y1'], [Variable('y2')], data_fidelity=(3,), name='comp2'))
    surr.fit()

    samples = surr.sample_inputs(10)
    pred = surr.predict(samples, use_model={'comp1': 'best'})

    assert all([pred['extrathing'][i].x == pred['extra'][i] for i in range(len(samples['x']))])
    assert 'more_extra' not in pred  # should not be returned since it is only using the surrogate for comp2
    assert np.allclose(pred['y1'], -samples['x']**2)  # should be exact
    assert relative_error(pred['y2'], 2*pred['y1']) < 1e-6


def test_invalid_sample_predict():
    """Test that prediction exits gracefully when a sample fails."""
    def faulty_model(inputs):
        if 0.4 < inputs['x'] < 0.6:
            raise Exception("This model is faulty... sorry:)")
        return {'y1': inputs['x'], 'y2': inputs['x'] ** 2}

    def also_faulty(inputs):
        if np.sqrt(inputs['y2']) < 0.1:
            raise Exception("You can't handle the truth")
        return {'y3': np.sqrt(inputs['y2'])}

    x = Variable(domain=(0, 1))
    comp1 = Component(faulty_model, [x], ['y1', 'y2'])
    comp2 = Component(also_faulty, ['y2'], ['y3'])
    system = System(comp1, comp2)

    samples = np.linspace(0, 1, 20)
    outputs = system.predict({'x': samples}, use_model='best')  # make sure nans carry through for faulty cases

    faulty_idx = ((samples > 0.4) & (samples < 0.6))

    for var in ['y1', 'y2']:
        assert np.all(np.isnan(outputs[var][faulty_idx]))
        assert np.all(~np.isnan(outputs[var][~faulty_idx]))

    faulty_idx |= (samples < 0.1)

    for var in ['y3']:
        assert np.all(np.isnan(outputs[var][faulty_idx]))
        assert np.all(~np.isnan(outputs[var][~faulty_idx]))

    assert all([faulty_idx[idx] for idx, err_info in enumerate(outputs['errors']) if err_info is not None])


def test_fire_sat(tmp_path, plots=False):
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

    # Test multiple interpolators on the Power component
    interpolators = {'lagrange': {},
                     'linear': {'regressor': 'RidgeCV', 'regressor_opts': {'alphas': np.logspace(-5, 4, 10).tolist()}},
                     'GPR': {'scaler': 'MinMaxScaler', 'kernel': 'PairwiseKernel', 'kernel_opts': {'metric': 'poly'}}
                     }
    surrogate_fidelities = {'lagrange': (), 'linear': (2,), 'GPR': ()}

    for interpolator, config in interpolators.items():
        surr.logger.info(f'Running "{interpolator}" interpolator...')
        surr.clear()
        surr['Power'].interpolator = Interpolator.from_dict(dict(method=interpolator, **config))
        surr['Power'].surrogate_fidelity = surrogate_fidelities[interpolator]

        with warnings.catch_warnings(action='ignore', category=(RuntimeWarning, ConvergenceWarning)):
            surr.fit(targets=targs, max_iter=10, max_tol=1e-3, runtime_hr=4/12, test_set=(xt, yt),
                     plot_interval=4, estimate_bounds=True)

        ysurr = surr.predict(xt, targets=targs)
        for var in yt:
            l2_error = relative_error(ysurr[var], yt[var])
            assert np.nanmax(l2_error) < 0.2

        # Plot 1d slices
        slice_idx = ['H', 'Po', 'Cd']
        try:
            fig, ax = surr.plot_slice(slice_idx, targs, show_model=['best', 'worst'], save_dir=surr.root_dir,
                                      random_walk=True, num_steps=15)
            fig.suptitle(f'{interpolator} interpolator')
        except np.linalg.LinAlgError:
            print('Its alright. Sometimes the random walks are wacky and FPI wont converge.')

        # Plot allocation bar charts
        surr.plot_allocation()

        if plots:
            # Plot error histograms
            fig, ax = plt.subplots(1, 3)
            kwargs = {'bins': 20, 'edgecolor': 'black', 'density': False, 'linewidth': 1.2}
            for i, var in enumerate(targs):
                ax[i].hist(yt[var], color='r', label='Truth', **kwargs)
                ax[i].hist(ysurr[var], color='b', alpha=0.4, label='Surrogate', **kwargs)
                ax_default(ax[i], surr.outputs()[var].get_tex(symbol=False, units=True), '', legend=True)

            fig.suptitle(f'{interpolator} interpolator')
            fig.set_size_inches(9, 3)
            fig.tight_layout()
            plt.show()


def test_turbojet_cycle(tmp_path):
    """Test for coupling a compressor, combuster, and turbine models. Includes a mix of analytical and surrogate
    models, as well as coupling field quantities along with scalar random variables.
    """
    def compressor(inputs):
        """Simple compressor model.

        :param inputs: `dict` of `Ta` and `v1` for ambient temperature and inlet velocity distribution
        :returns: `dict` of modified velocity `v2` and pressure distribution `p1`
        """
        inlet_velocity_distribution = inputs['v1']
        ambient_temp = inputs['Ta']
        compression_ratio = 1.5
        modified_velocity = inlet_velocity_distribution * compression_ratio
        modified_velocity += ambient_temp * 0.05
        base_pressure = 100
        pressure_boost_factor = 1.2
        pressure_distribution = base_pressure + (modified_velocity * pressure_boost_factor)
        return {'v2': modified_velocity, 'p1': pressure_distribution}

    def combuster(inputs, model_fidelity=None):
        """Simple combuster model.

        :param inputs: `dict` of `v2` and `Ta` for compressor outlet velocity and ambient temperature.
        :param model_fidelity: `tuple` of fidelity levels for combustion factor and temperature effect
        :returns: `dict` of outlet velocity `v3`
        """
        # Fidelity levels for combustion factor and external input effect
        alpha = model_fidelity or (3, 3)
        combustion_factor = 1.5 + 0.5 * alpha[0]
        temperature_factor = 0.05 + 0.03 * alpha[1]

        # Applying the combustion effect
        outlet_velocity = inputs['v2'] * combustion_factor + inputs['Ta'] * temperature_factor

        return {'v3': outlet_velocity}

    def turbine(inputs, model_fidelity=None):
        """Simple turbine model.

        :param inputs: `dict` of `v3` and `eta_t` for combuster outlet velocity and blade efficiency.
        :param model_fidelity: `tuple` of fidelity levels for turbine efficiency and other effects.
        :returns: `dict` of exit velocity `v4`
        """
        # Fidelity levels for turbine efficiency and other effects
        alpha = model_fidelity or (3,)
        turbine_efficiency = 0.7 + 0.1 * alpha[0]
        pressure_effect = 0.01 + 0.005 * alpha[0]
        efficiency_effect = 0.01 + 0.005 * alpha[0]

        # Applying the turbine efficiency effect
        exit_velocity_field = inputs['v3'] * turbine_efficiency

        # Adjustments based on pressure field and external input effects
        blade_efficiency = inputs['eta_t']
        exit_velocity_field += inputs['p1'] * pressure_effect
        exit_velocity_field += efficiency_effect * blade_efficiency

        return {'v4': exit_velocity_field}

    # Synthetic velocity distributions
    num_samples = 100
    num_points = 80
    inlet_grid = np.linspace(-1, 1, num_points)
    means = np.random.uniform(200, 300, num_samples)
    stds = np.random.uniform(0.15, 0.35, num_samples)
    v1_samples = np.zeros((num_points, num_samples))
    for i in range(num_samples):
        v1_samples[:, i] = means[i] * np.exp(-0.5 * (inlet_grid / stds[i]) ** 2)

    # Random variables
    Ta = Variable('Ta', distribution='N(300, 20)', units='K', description='Ambient temperature', norm='zscore')
    eta_t = Variable('eta_t', distribution='N(0.8, 0.04)', description='Turbine blade efficiency')
    Ta_samples = Ta.sample(num_samples)
    eta_t_samples = eta_t.sample(num_samples)

    # Field quantities
    v1 = Variable('v1', units='m/s', description='Inlet velocity', compression=SVD(rank=4, coords=inlet_grid))
    v2 = Variable('v2', units='m/s', description='Compressor outlet velocity',
                  compression=SVD(rank=3, coords=inlet_grid))
    v3 = Variable('v3', units='m/s', description='Combuster outlet velocity',
                  compression=SVD(rank=2, coords=inlet_grid))
    v4 = Variable('v4', units='m/s', description='Turbine exit velocity',
                  compression=SVD(rank=4, coords=inlet_grid))
    p1 = Variable('p1', units='kPa', description='Compressor outlet pressure',
                  compression=SVD(rank=2, coords=inlet_grid))

    # Define components
    comp1 = Component(compressor, [Ta, v1], [v2, p1], name='Compressor')
    comp2 = Component(combuster, [Ta, v2], [v3], data_fidelity=(2, 2), model_fidelity=(3, 3), name='Combuster')
    comp3 = Component(turbine, [eta_t, v3, p1], [v4], data_fidelity=(2, 2, 2), model_fidelity=(3,), name='Turbine')
    system = System(comp1, comp2, comp3, root_dir=tmp_path)

    def _object_to_numeric(array):  # for field quantities
        return np.concatenate([arr[np.newaxis, ...] for arr in array], axis=0)

    # Generate compression data
    xtest = {'Ta': Ta_samples, 'v1': v1_samples.T, 'eta_t': eta_t_samples}
    ytest = system.predict(xtest, use_model='best', normalized_inputs=False)
    v1.compression.compute_map(data_matrix=v1_samples)
    v2.compression.compute_map(data_matrix=_object_to_numeric(ytest['v2']).T)
    v3.compression.compute_map(data_matrix=_object_to_numeric(ytest['v3']).T)
    v4.compression.compute_map(data_matrix=_object_to_numeric(ytest['v4']).T)
    p1.compression.compute_map(data_matrix=_object_to_numeric(ytest['p1']).T)

    # Set domain estimate for v1 latent coefficients
    v1_latent = v1.compression.compress(v1_samples.T)
    v1.update_domain(list(zip(np.min(v1_latent, axis=0), np.max(v1_latent, axis=0))), override=True)

    system.fit(max_iter=15, test_set=(xtest, ytest), estimate_bounds=True)

    # Check error
    final_iter = system.train_history[-1]
    for var, perf in final_iter['test_error'].items():
        assert perf < 0.2
