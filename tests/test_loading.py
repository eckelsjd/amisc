import numpy as np
import os
from pathlib import Path
import shutil
import time

from amisc.system import SystemSurrogate, ComponentSpec
from amisc.rv import UniformRV


def coupled_system(save_dir=None, logger_name=None):
    def f1(x):
        return {'y': x * np.sin(np.pi * x)}
    def f2(x):
        return {'y': 1 / (1 + 25 * x ** 2)}

    exo_vars = [UniformRV(0, 1)]
    coupling_vars = [UniformRV(0, 1), UniformRV(0, 1)]
    comp1 = ComponentSpec(f1, name='Model1', exo_in=0, coupling_out=0)
    comp2 = ComponentSpec(f2, name='Model2', coupling_in=0, coupling_out=1)
    surr = SystemSurrogate([comp1, comp2], exo_vars, coupling_vars, init_surr=True, logger_name=logger_name,
                           save_dir=save_dir)
    return surr


def test_save_load():
    """Test saving and loading from .pkl file."""
    surr = coupled_system(save_dir=None)
    surr.fit(max_iter=2)
    surr.save_to_file('test_surrogate.pkl')

    surr2 = SystemSurrogate.load_from_file('test_surrogate.pkl')
    surr2.fit(max_iter=2)
    os.unlink('test_surrogate.pkl')

    assert surr2.log_file is None and surr2.root_dir is None

    for f in os.listdir(Path('.')):
        if f.startswith('amisc_') or f.endswith('.pkl'):
            assert False


def test_save_dir():
    """Test saving and loading from amisc_timestamp directory"""
    surr = coupled_system(save_dir='.')
    surr.fit(max_iter=3, save_interval=1)
    surr.save_to_file('test_surrogate.pkl')

    for f in os.listdir(Path('.')):
        if f.startswith('amisc_'):
            amisc_dir = Path('.') / f
            surr2 = SystemSurrogate.load_from_file(amisc_dir / 'sys' / 'test_surrogate.pkl')
            assert len(os.listdir(amisc_dir / 'sys')) == 6
            assert surr2.log_file == surr.log_file

    for f in os.listdir(Path('.')):
        if f.endswith('.pkl'):
            assert False


def test_two_surrogates():
    """Test logging with two surrogates at the same time."""
    surr1 = coupled_system(save_dir='.', logger_name='Surrogate 1')
    time.sleep(1.5)
    surr2 = coupled_system(save_dir='.', logger_name='Surrogate 2')

    surr1.fit(max_iter=2)
    surr2.fit(max_iter=2)

    assert surr1.log_file != surr2.log_file
