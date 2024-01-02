import numpy as np

from amisc.examples.models import borehole_func, wing_weight_func
from amisc.system import SystemSurrogate, ComponentSpec
from amisc.rv import UniformRV


def converge(system='Borehole', N=1000):
    """Test convergence of different model surrogates."""
    sys = None
    match system:
        case 'Borehole':
            exo_vars = [UniformRV(0.05, 0.15, 'rw'), UniformRV(100, 50000, 'r'), UniformRV(63070, 115600, 'Tu'),
                        UniformRV(990, 1110, 'Hu'), UniformRV(63.1, 116, 'Tl'), UniformRV(700, 820, 'Hl'),
                        UniformRV(1120, 1680, 'L'), UniformRV(9855, 12045, 'Kw')]
            coupling_vars = [UniformRV(0, 1000, 'vdot')]
            sys = SystemSurrogate(ComponentSpec(borehole_func, name=system), exo_vars, coupling_vars, est_bds=1000,
                                  stdout=False)
        case 'Wing':
            exo_vars = [UniformRV(150, 250, id='Sw'), UniformRV(220, 300, id='Wfw'), UniformRV(6, 10, id='A'),
                        UniformRV(-10, 10, id='Lambda'), UniformRV(16, 45, id='q'), UniformRV(0.5, 1, id='lambda'),
                        UniformRV(0.08, 0.18, id='tc'), UniformRV(2.5, 6, id='Nz'), UniformRV(1700, 2500, id='Wdg'),
                        UniformRV(0.025, 0.08, id='Wp')]
            coupling_vars = [UniformRV(0, 10000, id='Wwing')]
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
