from pathlib import Path

import numpy as np
import yaml

from amisc.examples.models import borehole_func, wing_weight_func
from amisc.system import ComponentSpec, SystemSurrogate


def converge(system='Borehole', N=1000):
    """Test convergence of different model surrogates."""
    sys = None
    match system:
        case 'Borehole':
            with open(Path(__file__).parent / 'borehole.yml', 'r') as fd:
                data = yaml.load(fd, yaml.Loader)
                exo_vars = data['exo']
                coupling_vars = data['coupling']
            sys = SystemSurrogate(ComponentSpec(borehole_func, name=system), exo_vars, coupling_vars, est_bds=1000,
                                  stdout=False)
        case 'Wing':
            with open(Path(__file__).parent / 'wing.yml', 'r') as fd:
                data = yaml.load(fd, yaml.Loader)
                exo_vars = data['exo']
                coupling_vars = data['coupling']
            sys = SystemSurrogate(ComponentSpec(wing_weight_func, name=system), exo_vars, coupling_vars, est_bds=1000,
                                  stdout=False)

    # Random test set for percent error
    xt = sys.sample_inputs((N,))
    yt = sys(xt, use_model='best')

    sys.fit()
    ysurr = sys.predict(xt)
    l2_error = np.sqrt(np.mean((yt-ysurr)**2, axis=0)) / np.sqrt(np.mean(yt**2, axis=0))

    return l2_error


def test_borehole(tol=1e-2):
    l2_error = converge(system='Borehole')
    assert np.max(l2_error) < tol


def test_wing_weight(tol=1e-2):
    l2_error = converge(system='Wing')
    assert np.max(l2_error) < tol
