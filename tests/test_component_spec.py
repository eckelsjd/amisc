"""Test that component specification and validation works as expected using pydantic."""
import itertools
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from amisc import yaml_dump, yaml_load
from amisc.component import (
    Component,
    IndexSet,
    Interpolator,
    InterpolatorState,
    LagrangeState,
    MiscTree,
    ModelArgs,
    ModelKwargs,
    TrainingData,
)
from amisc.serialize import Base64Serializable, Serializable, StringSerializable
from amisc.variable import Variable


def test_model_args_kwargs():
    # Test ModelArgs
    cases = [(1, 2, 3), (1.1, -1.3e6, True), ('hello', '12', 'False', (2, 2)),
             (None, [1, 2, 3], {'key': 'value'}), (3.14, 2.718, 1.618), (True, False, None)]
    for case in cases:
        args = ModelArgs(*case)
        serialized_args = args.serialize()
        deserialized_args = ModelArgs.deserialize(serialized_args)
        assert deserialized_args.data == args.data

    # Test ModelKwargs
    cases = [{'a': 1, 'b': 2, 'c': 3}, {'a': 1.1, 'b': -1.3e6, 'c': True},
             {'hello': False, 'False': (2, 2), '1': '2'},
             {'x': None, 'y': [1, 2, 3], 'z': {'key': 'value'}},
             {'pi': 3.14, 'e': 2.718, 'phi': 1.618},
             {'flag1': True, 'flag2': False, 'flag3': None},
             {'nested': {'level1': {'level2': {'level3': 'deep_value', 'level4': [True, False, (1, 2, 3)]}}}}]
    for case in cases:
        kwargs = ModelKwargs(**case)
        serialized_kwargs = kwargs.serialize()
        deserialized_kwargs = ModelKwargs.deserialize(serialized_kwargs)
        assert deserialized_kwargs.data == kwargs.data


def test_indexset():
    # Test IndexSet data structure
    Ik = [(np.random.randint(0, 10, 2), np.random.randint(0, 5, 4)) for i in range(10)]
    new_indices = [((), np.random.rand(3)*10) for i in range(5)]
    Ik = IndexSet(*Ik)
    Ik.extend(new_indices)
    del Ik[0]
    Ik[0] = ((), (1.1, 2.34))
    Ik.insert(1, ((1,), (-1.1, 2.3)))
    Ik.append(((1,), (1, 2, 3)))
    for alpha, beta in Ik:
        assert np.all([isinstance(ele, int) for ele in itertools.chain(alpha, beta)])
    Ik.data = [str(([int(j) for j in np.random.randint(-100, 100, 7)],
                    [int(j) for j in -np.random.rand(2)*3])) for i in range(3)]
    for alpha, beta in Ik:
        assert np.all([isinstance(ele, int) for ele in itertools.chain(alpha, beta)])


def test_misctree():
    # Test MiscTree data structure
    data = {tuple([int(k) for k in np.random.randint(0, 3, 2)]) :
                {tuple([int(k) for k in -np.random.rand(3)*10 + 5]): np.random.rand() for i in range(5)}
            for j in range(2)}
    new_data = {'()': {f'({i},)': np.random.rand() for i in range(4)}}
    tree = MiscTree(data=data)
    tree[(), (0,)] = 100
    assert tree.get(('()', '(0,)')) == 100
    tree.update(new_data)
    for alpha, beta, data in tree:
        assert np.all([isinstance(ele, int) for ele in itertools.chain(alpha, beta)])

    # Test serialization of InterpolatorState
    data = {(j, 3): {(i,): LagrangeState(weights=[np.random.rand(3)], x_grids=[np.random.rand(3)]) for i in range(4)}
            for j in range(2)}
    tree = MiscTree(data)
    serialized_tree = tree.serialize()
    deserialized_tree = MiscTree.deserialize(serialized_tree)
    assert deserialized_tree == tree


def simple_model(x, alpha, error=0.1, output_path='.'):
    return alpha * error + x ** 2


def test_component_validation(tmp_path):
    # Test completely basic component init
    def my_model(x):
        return x
    inputs = [Variable() for i in range(3)]
    outputs = [Variable() for i in range(3)]
    comp = Component(my_model, inputs, outputs)
    assert len(comp.active_set) + len(comp.candidate_set) == 0
    assert comp.xdim + comp.ydim == len(inputs) + len(outputs)
    for attribute in ['model_args', 'model_kwargs', 'misc_states', 'misc_costs', 'misc_coeff', 'interpolator',
                      'training_data']:
        assert isinstance(getattr(comp, attribute), Serializable)

    # Test partial validation
    variables = [Variable(dist=f'U(0, {np.random.rand()})') for i in range(10)]
    alphas = [(), (0,), (1,)]
    betas = [(0, 0), (0., 1.1), (1, 0)]

    active_set = []
    misc_states = {}
    misc_costs = {}
    for alpha in alphas:
        for beta in betas:
            active_set.append((alpha, beta))
            misc_costs.setdefault(alpha, dict())
            misc_states.setdefault(alpha, dict())
            misc_costs[alpha][beta] = 1.0 if np.random.rand() < 0.5 else -1.0
            misc_states[alpha][beta] = LagrangeState(weights=[np.random.rand(3)], x_grids=[np.random.rand(3)])

    c = Component(simple_model, variables[:5], outputs=variables[5:], model_args=(1,),
                  model_kwargs={'error': 0.2}, max_alpha=(2,), max_beta=(3, 2),
                  active_set=active_set, misc_states=misc_states, misc_costs=misc_costs)
    serialize_kwargs = {'training_data': {'save_path': tmp_path / 'training_data.pkl'}}
    c2 = c.serialize(serialize_kwargs=serialize_kwargs)
    c3 = Component.deserialize(c2)
    assert c3 == c


class CustomArgs(ModelArgs):
    def __init__(self, number, *args):
        self.number = number
        super().__init__(*args)

    def __str__(self):
        return f'CustomArgs({", ".join([str(self.number)] + [f"{value}" for value in self.data])})'


@dataclass
class CustomInterpolator(Interpolator, Base64Serializable):
    kernel_width: float = 0.1
    kernel_type: str = 'gaussian'


@dataclass
class CustomInterpolatorState(InterpolatorState, StringSerializable):
    knots: list[ArrayLike, ...] = field(default_factory=list)


@dataclass
class CustomDataStorage(TrainingData, StringSerializable):
    length: int = 40
    width: int = 2


def test_custom_data_classes():
    # Test custom args/kwargs, states, interpolator, etc.
    c = Component(simple_model, Variable(), Variable(), interpolator=CustomInterpolator(),
                  training_data=CustomDataStorage(), model_args=CustomArgs(100),
                  misc_states={(): {(): CustomInterpolatorState()}})
    c2 = c.serialize()
    c3 = Component.deserialize(c2)
    assert c3 == c


def test_save_and_load(tmp_path):
    # Test loading from yaml
    alphas = [(), (0,), (1,)]
    betas = [(0, ), (0, 0), (0, 1)]

    active_set = []
    misc_states = {}
    misc_costs = {}
    for alpha in alphas:
        for beta in betas:
            active_set.append((alpha, beta))
            misc_costs.setdefault(alpha, dict())
            misc_states.setdefault(alpha, dict())
            misc_costs[alpha][beta] = 1.0 if np.random.rand() < 0.5 else -1.0
            misc_states[alpha][beta] = LagrangeState(weights=[np.random.rand(5)], x_grids=[np.random.rand(5)])
    c = Component(simple_model, [Variable() for i in range(3)], [Variable() for i in range(3)],
                  active_set=active_set, misc_costs=misc_costs, misc_states=misc_states, model_args=(1, 'hello', 3.14),
                  model_kwargs={'output_path': '.', 'opts': {'max_iter': 1000}}, max_alpha=(3,), status=1)

    savepath = tmp_path / 'component.yml'
    yaml_dump(c, savepath)
    c_load = yaml_load(savepath)
    assert c_load == c


@dataclass
class Extra:
    x1: float


def special_model(x, name, frac, alpha=(1,), output_path='.', output_vars=None):
    t1 = time.time()
    ret_dict = {'y1': x['x1'] + x['x2'], 'y2': x['x2'] * x['x3']}
    if np.random.rand() < frac:
        raise ValueError(f'Random error: {uuid.uuid4().hex}')
    mult = 1 if name == 'hello' else 2.5
    err = mult * (1 / alpha[0])
    if output_vars is not None:
        err += output_vars['y1'].get_nominal()
    ret_dict['y3'] = np.ones((15, 2)) + err
    filename = f'output_{uuid.uuid4().hex}.txt'
    with open(Path(output_path) / filename, 'w') as fd:
        np.savetxt(fd, np.random.rand(10, 3))
    ret_dict['model_cost'] = np.random.rand() + (time.time() - t1)
    ret_dict['output_path'] = filename
    ret_dict['extra_ret'] = Extra(x1=x['x1'])
    return ret_dict


def test_model_wrapper(tmp_path):
    # Vectorized call
    def vectorized_model(x):
        return {
            'y1': x['x1'] + x['x2'],
            'y2': x['x2'] * x['x3'],
            'y3': np.ones((15, 2)) * x['x1'][..., np.newaxis, np.newaxis]
        }

    inputs = [Variable(var_id='x1'), Variable(var_id='x2'), Variable(var_id='x3')]
    outputs = [Variable(var_id='y1', nominal=0.1), Variable(var_id='y2'), Variable(var_id='y3')]
    comp = Component(vectorized_model, inputs, outputs, vectorized=True)

    cases = [{'x1': np.random.rand(10), 'x2': np.random.rand(10), 'x3': np.random.rand(10)},
             {'x1': np.random.rand(3, 5), 'x2': np.random.rand(3, 5), 'x3': np.random.rand(3, 5)},
             {'x1': np.random.rand(), 'x2': np.random.rand(), 'x3': np.random.rand()},
             [np.random.rand() for i in range(3)]]
    expected = [vectorized_model(cases[i]) for i in range(2)]
    expected += [vectorized_model({key: np.atleast_1d(cases[2][key]) for key in ['x1', 'x2', 'x3']})]
    expected += [vectorized_model({key: np.atleast_1d(cases[3][i]) for i, key in enumerate(['x1', 'x2', 'x3'])})]
    for i, case in enumerate(cases):
        y = comp.call_model(case)
        for key, val in y.items():
            assert np.allclose(val, expected[i][key])

    # Serial call
    def serial_model(x):
        return {
            'y1': float(x['x1'] + x['x2']),
            'y2': float(x['x2'] * x['x3']),
            'y3': np.ones((15, 2)) * x['x1']
        }
    comp = Component(serial_model, inputs, outputs)
    for i, case in enumerate(cases):
        y = comp.call_model(case)
        for key, val in y.items():
            assert np.allclose(val, expected[i][key])

    # Parallel call
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, case in enumerate(cases):
            y = comp.call_model(case, executor=executor)
            for key, val in y.items():
                assert np.allclose(val, expected[i][key])

    # Extra features (alpha, output_path, vars, exceptions, etc.)
    with ProcessPoolExecutor(max_workers=4) as executor:
        comp = Component(special_model, inputs, outputs, max_alpha=(2,), model_args=('hello', 0.2), executor=executor)
        shape = (5, 2, 2)
        case = {'x1': np.random.rand(*shape), 'x2': np.random.rand(*shape), 'x3': np.random.rand(*shape)}
        ret = comp.call_model(case, alpha='best', output_path=tmp_path)
    assert ret['y3'].shape == shape + (15, 2)
    for key in ['y1', 'y2', 'extra_ret', 'model_cost', 'output_path']:
        assert ret[key].shape == shape
    y1_ravel = np.ravel(ret['y1'])
    y2_ravel = np.ravel(ret['y2'])
    y3_ravel = np.reshape(ret['y3'], (-1, 15, 2))
    if ret.get('errors'):
        for idx, error in ret.get('errors').items():
            assert 'Random error' in error['error']
            assert np.isnan(y1_ravel[idx])
            assert np.isnan(y2_ravel[idx])
            assert np.all(np.isnan(y3_ravel[idx, ...]))
