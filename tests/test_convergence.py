"""Test convergence of `System.fit` to a few smooth analytical models."""
from pathlib import Path
from typing import Literal

import numpy as np

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
