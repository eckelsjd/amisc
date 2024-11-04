"""Test that the [`SystemSurrogate`][amisc.system.SystemSurrogate] class validation works."""
# ruff: noqa: E741
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from amisc import Component, System, Variable, YamlLoader
from amisc.system import TrainHistory


def simple_model(inputs, output_path='.'):
    return {'y': inputs['x'] ** 2}


def test_basic_init(tmp_path):
    """Test inputs/output/graph and root directory/logger/executor initialization."""
    comp1 = Component(simple_model, [Variable(l) for l in 'abc'], [Variable(l) for l in 'def'], name='comp1')
    comp2 = Component(simple_model, [Variable(l) for l in 'dghi'], [Variable(l) for l in 'jkl'], name='comp2')
    comp3 = Component(simple_model, [Variable(l) for l in 'lmno'], [Variable(l) for l in 'pqra'], name='comp3')
    system = System(comp1, [comp2], comp3)
    assert list(system.graph().edges()) == [('comp1', 'comp2'), ('comp2', 'comp3'), ('comp3', 'comp1')]
    assert system.inputs().keys() == {'b', 'c', 'g', 'h', 'i', 'm', 'n', 'o'}
    assert system.outputs().keys() == {'d', 'e', 'f', 'j', 'k', 'l', 'p', 'q', 'r', 'a'}

    paths = [tmp_path, tmp_path / 'amisc_test']
    for p in paths:
        s1 = System(comp1, comp2, comp3, root_dir=p)
        assert s1.root_dir.name.startswith('amisc_')
        assert (s1.root_dir / 'surrogates').is_dir()
        for comp in s1.components:
            assert (s1.root_dir / 'components' / comp.name).is_dir()
            assert comp.logger.handlers[0].baseFilename == s1.logger.handlers[0].baseFilename

    serialize_kwargs = {comp.name: {'training_data': {'save_path': tmp_path / f'{comp.name}_training_data.pkl'}}
                        for comp in s1.components}
    assert System.deserialize(s1.serialize(serialize_kwargs=serialize_kwargs)) == s1


def test_init_methods():
    """Test convenience methods for specifying a model."""
    # Call/return unpacked
    def my_model(x1, x2, model_fidelity=(1,)):
        y1 = x1 ** 2
        y2 = x2 ** 3
        return y1, y2
    s = System(my_model)
    inputs = {'x1': np.linspace(0, 1, 50), 'x2': np.linspace(0, 1, 50)}
    outputs = s.predict(inputs, use_model='best')
    expected = my_model(**inputs)
    assert np.allclose(outputs['y1'], expected[0])
    assert np.allclose(outputs['y2'], expected[1])
    assert s['my_model'].call_unpacked and s['my_model'].ret_unpacked
    assert all([v in s['my_model'].inputs for v in ['x1', 'x2']])
    assert all([v in s['my_model'].outputs for v in ['y1', 'y2']])

    # Call unpacked, return packed
    def my_model(x1, x2):
        return {'y1': x1 ** 2, 'y2': x2 ** 2}
    s = System(Component(my_model, Variable('x1', domain=(1, 4)), Variable('x2', domain=(0, 1)), 'y1', 'y2'))
    inputs = s.sample_inputs(10)
    outputs = s.predict(inputs, use_model='best')
    expected = my_model(inputs['x1'], inputs['x2'])
    assert np.allclose(outputs['y1'], expected['y1'])
    assert np.allclose(outputs['y2'], expected['y2'])
    assert s['my_model'].call_unpacked and not s['my_model'].ret_unpacked

    # Call packed, return unpacked
    def my_model(inputs):
        y1 = inputs['x1'] ** 2
        y2 = inputs['x2'] ** 2
        return y1, y2
    s = System(Component(my_model, [Variable('x1', domain=(-1, 1)), Variable('x2', domain=(-1, 1))]))
    inputs = s.sample_inputs((10, 2))
    outputs = s.predict(inputs, use_model='best')
    expected = my_model(inputs)
    assert np.allclose(outputs['y1'], expected[0])
    assert np.allclose(outputs['y2'], expected[1])
    assert not s['my_model'].call_unpacked and s['my_model'].ret_unpacked

    # Call/return packed
    def my_model(inputs):
        return {'y1': inputs['x1'] ** 2, 'y2': inputs['x2'] ** 2}
    s = System(Component(my_model, [Variable('x1', domain=(-1, 1)), Variable('x2', domain=(-1, 1))], ['y1', 'y2']))
    inputs = s.sample_inputs((20, 1))
    outputs = s.predict(inputs, use_model='best')
    expected = my_model(inputs)
    assert np.allclose(outputs['y1'], expected['y1'])
    assert np.allclose(outputs['y2'], expected['y2'])
    assert not s['my_model'].call_unpacked and not s['my_model'].ret_unpacked

    # Assert warning when trying unpacked usage of single input/output model
    def my_model(x1):
        y1 = x1 ** 2
        return y1
    with warnings.catch_warnings(record=True) as w:
        s = System(Component(my_model, Variable('x1'), Variable('y1')))
        assert len(w) == 2
        assert issubclass(w[0].category, UserWarning)
        assert issubclass(w[1].category, UserWarning)
        assert 'If you intended to use a single input argument' in str(w[0].message)
        assert 'If you intended to output a single return value' in str(w[1].message)
        assert not s['my_model'].call_unpacked and not s['my_model'].ret_unpacked

    # Assert value error raised if missing inputs or outputs
    def my_model(inputs):
        return {'y1': inputs['x1'] ** 2}
    with pytest.raises(ValueError):
        s = System(Component(my_model, inputs=None, outputs=None))
    with pytest.raises(ValueError):
        s2 = System(lambda: None)  # noqa: F841


def test_save_and_load(tmp_path):
    save_file = tmp_path / 'system.yml'
    s1 = System(Component(simple_model, Variable('x'), Variable('y'), name='comp1'))
    YamlLoader.dump(s1, save_file)
    os.mkdir(tmp_path / 'new_path')
    shutil.move(save_file, tmp_path / 'new_path' / save_file.name)
    save_file = tmp_path / 'new_path' / save_file.name
    for f in os.listdir(tmp_path):
        if Path(f).suffix == '.pkl':
            shutil.move(tmp_path / f, tmp_path / 'new_path' / f)
    os.chdir(tmp_path / 'new_path')  # change directory to test that yaml load still finds the .pkl save files
    s2 = YamlLoader.load(save_file)
    assert s2 == s1


def test_empty_inspection():
    """Load a completely empty system. Get variables and components from inspection."""
    def first(x):
        y = x ** 2
        return y

    def second(x, y):
        z = x + y
        z2 = y ** 2
        return z, z2
    system = System(first, second)
    assert all([comp.call_unpacked for comp in system.components])
    assert all([var in ['x'] for var in system.inputs()])
    assert all([var in ['y', 'z', 'z2'] for var in system.outputs()])
    assert system['first'].inputs['x'].name == 'x'
    assert system['first'].outputs['y'].name == 'y'
    assert system['first'].inputs['x'] is system['second'].inputs['x']
    assert system['first'].outputs['y'] is system['second'].inputs['y']

    x_grid = np.linspace(0, 1, 100)
    y_grid = system.predict({'x': x_grid}, use_model='best')
    assert np.allclose(first(x_grid), y_grid['y'])
    assert np.allclose(second(x_grid, first(x_grid))[0], y_grid['z'])
    assert np.allclose(second(x_grid, first(x_grid))[1], y_grid['z2'])


def test_swap_and_insert():
    """Test swapping and inserting a new component into a system."""
    def first(x):
        y = x ** 2
        return y

    def first_swap(x):
        y = x ** 3
        y2 = x ** 2 / 2
        return y, y2

    def second(x, y):
        z = x + y
        z2 = y ** 2
        return z, z2

    def third(z, z2):
        z3 = z + z2
        return z3

    system = System(first, second)
    system.insert_components(third)
    system.swap_component('first', first_swap)
    assert all([comp.name in ['first_swap', 'second', 'third'] for comp in system.components])
    assert all([var in system.inputs() for var in ['x']])
    assert all([var in system.outputs() for var in ['y', 'y2', 'z', 'z2', 'z3']])

    system.remove_component('third')
    assert len(system.components) == 2
    assert system.components[0].name == 'first_swap'
    assert system.components[1].name == 'second'


def test_train_history():
    # Test TrainHistory data structure
    hist = [{'component': f'comp{i}', 'alpha': tuple(np.random.randint(0, 10, 3)),
             'beta': tuple(np.random.randint(0, 10, 2)), 'num_evals': np.random.rand(),
             'added_cost': np.random.rand(), 'added_error': np.random.rand(), 'test_error': {'y': np.random.rand()}}
            for i in range(10)]
    new_hist = [{'component': f'comp{int(np.random.randint(10))}', 'alpha': tuple(np.random.randint(0, 3, 2)),
                 'beta': tuple(np.random.randint(0, 5, 4)), 'num_evals': np.random.rand(),
                 'added_cost': np.random.rand(), 'added_error': np.random.rand()} for i in range(5)]
    hist = TrainHistory(data=hist)
    hist.extend(new_hist)
    assert TrainHistory.deserialize(hist.serialize()) == hist


def test_save_dir(tmp_path):
    """Test saving and loading from `amisc_timestamp` directory"""
    comp = Component(simple_model, Variable('x', domain=(0, 1)), Variable('y'), name='comp1', data_fidelity=(4,))
    surr = System(comp, root_dir=tmp_path, name='test_system')
    surr.fit(max_iter=3, save_interval=1)
    surr.save_to_file('test_surrogate.yml')

    found_dir = False
    for f in os.listdir(tmp_path):
        if f.startswith('amisc_'):
            amisc_dir = tmp_path / f
            surr2 = System.load_from_file(amisc_dir / 'surrogates' / 'test_surrogate.yml')
            assert (amisc_dir / 'components' / 'comp1').is_dir()
            assert all([(amisc_dir / 'surrogates' / f'{surr.name}_iter{i}' / f'{surr.name}_iter{i}.yml').exists()
                        for i in range(1, 4)])
            assert surr2 == surr
            found_dir = True

    assert found_dir
