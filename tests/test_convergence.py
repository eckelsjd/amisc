"""Test convergence of `System.fit` to a few smooth analytical models."""
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from amisc import Component, Variable
from amisc.interpolator import Interpolator
from amisc.system import System
from amisc.utils import relative_error


def converge(system: Literal['borehole', 'wing'] = 'borehole', num_samples=1000):
    """Test convergence of different model surrogates."""
    surr = System.load_from_file(Path(__file__).parent / f'{system.lower()}.yml')

    xt = surr.sample_inputs(num_samples)
    yt = surr.predict(xt, use_model='best')

    surr.fit()
    ysurr = surr.predict(xt)
    l2_error = {var: relative_error(ysurr[var], yt[var]) for var in yt}

    return l2_error


def test_borehole(tol=1e-2):
    l2_error = converge(system='borehole')
    assert np.max(l2_error['vdot']) < tol


def test_wing_weight(tol=1e-2):
    l2_error = converge(system='wing')
    assert np.max(l2_error['Wwing']) < tol


def test_curved_with_noise():
    """Noisy function convergence."""
    def curved_func(inputs, noise_std=0.0):
        """From https://www.sfu.ca/~ssurjano/detpep10curv.html"""
        x1, x2, x3 = inputs['x1'], inputs['x2'], inputs['x3']
        y = 4 * (x1 - 2 + 8 * x2 - 8 * x2 ** 2) ** 2 + (3 - 4 * x2) ** 2 + 16 * np.sqrt(x3 + 1) * (2 * x3 - 1) ** 2

        if noise_std > 0:
            y += np.random.normal(0, noise_std, y.shape)

        return {'y': y}

    noise_std = 2
    linear_opts = {'regressor': 'RidgeCV', 'regressor_opts': {'alphas': np.logspace(-8, -1, 9).tolist()},
                   'polynomial_opts': {'degree': 3},
                   'scaler': 'MinMaxScaler', 'scaler_opts': {'feature_range': (-1, 1)}}
    x1 = Variable(distribution='U(0, 1)')
    x2 = Variable(distribution='U(0, 1)')
    x3 = Variable(distribution='U(0, 1)')
    system = System(Component(curved_func, inputs=[x1, x2, x3], outputs=['y'], vectorized=True, data_fidelity=(3, 3, 3),
                              interpolator=Interpolator.from_dict(dict(method='linear', **linear_opts)),
                              noise_std=noise_std))

    xt = system.sample_inputs(1000)
    yt = curved_func(xt)

    with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
        system.fit(max_iter=50, max_tol=1e-4)
    ysurr = system.predict(xt)
    l2_error = relative_error(ysurr['y'], yt['y'])
    rel_noise = 2 * noise_std / np.mean(yt['y'])
    assert l2_error < rel_noise, f"L2 error: {l2_error} is greater than relative noise level: {rel_noise}"
