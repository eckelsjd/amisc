"""Test that component specification and validation works as expected using pydantic. Also test component
MISC evaluation with sparse grids.
"""
import itertools
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from uqtils import approx_hess, approx_jac, ax_default

from amisc import YamlLoader
from amisc.component import Component, IndexSet, MiscTree, ModelKwargs, StringKwargs
from amisc.compression import SVD
from amisc.interpolator import Interpolator, InterpolatorState, LagrangeState, Linear
from amisc.serialize import Base64Serializable, Serializable, StringSerializable
from amisc.training import TrainingData
from amisc.typing import LATENT_STR_ID
from amisc.utils import relative_error, to_model_dataset, to_surrogate_dataset
from amisc.variable import Variable


def test_model_kwargs():
    cases = [{'a': 1, 'b': 2, 'c': 3}, {'a': 1.1, 'b': -1.3e6, 'c': True},
             {'hello': False, 'False': (2, 2), '1': '2'},
             {'x': None, 'y': [1, 2, 3], 'z': {'key': 'value'}},
             {'pi': 3.14, 'e': 2.718, 'phi': 1.618},
             {'flag1': True, 'flag2': False, 'flag3': None},
             {'nested': {'level1': {'level2': {'level3': 'deep_value', 'level4': [True, False, (1, 2, 3)]}}}}]
    for case in cases:
        kwargs = ModelKwargs(**case)
        str_kwargs = ModelKwargs.from_dict({'method': 'string_kwargs', **case})
        serialized_kwargs = kwargs.serialize()
        str_serial = str_kwargs.serialize()
        deserialized_kwargs = ModelKwargs.deserialize(serialized_kwargs)
        str_deserialized = StringKwargs.deserialize(str_serial)
        assert deserialized_kwargs.data == kwargs.data
        assert str_deserialized.data == kwargs.data


def test_indexset():
    # Test IndexSet data structure
    Ik = [(np.random.randint(0, 10, 2), np.random.randint(0, 5, 4)) for i in range(10)]
    new_indices = [((), np.random.rand(3)*10) for i in range(5)]
    Ik = IndexSet(Ik)
    Ik.update(new_indices)
    Ik.add(((), (1.1, 2.34)))
    Ik.remove(((), (1, 2)))
    Ik.add(((1,), (-1.1, 2.3)))
    Ik = Ik.union({((1,), (1, 2, 3))})
    for alpha, beta in Ik:
        assert np.all([isinstance(ele, int) for ele in itertools.chain(alpha, beta)])
    Ik = IndexSet([str(([int(j) for j in np.random.randint(-100, 100, 7)],
                        [int(j) for j in -np.random.rand(2)*3])) for i in range(3)])
    for alpha, beta in Ik:
        assert np.all([isinstance(ele, int) for ele in itertools.chain(alpha, beta)])

    assert IndexSet.deserialize(Ik.serialize()) == Ik


def test_misctree():
    # Test MiscTree data structure
    data = {tuple([int(k) for k in np.random.randint(0, 3, 2)]):
                {tuple([int(k) for k in -np.random.rand(3)*10 + 5]): np.random.rand() for i in range(5)}
            for j in range(2)}
    new_data = {'()': {f'({i},)': np.random.rand() for i in range(4)}}
    tree = MiscTree(data)
    tree[(), (0,)] = 100
    assert tree.get(('()', '(0,)')) == 100
    tree.update(new_data)
    for alpha, beta, data in tree:
        assert np.all([isinstance(ele, int) for ele in itertools.chain(alpha, beta)])

    # Test serialization of InterpolatorState
    data = {(j, 3): {(i,): LagrangeState(weights={'x': np.random.rand(3)}, x_grids={'x': np.random.rand(3)})
                     for i in range(4)} for j in range(2)}
    tree = MiscTree(data)
    assert MiscTree.deserialize(tree.serialize()) == tree


def simple_model(x, model_fidelity=(1,), error=0.1, output_path='.'):
    return model_fidelity[0] * error + x ** 2


def test_component_validation(tmp_path):
    # Test basic component init
    def my_model(x):
        return x
    inputs = [Variable() for i in range(3)]
    outputs = [Variable() for i in range(3)]
    comp = Component(my_model, inputs, outputs)
    assert len(comp.active_set) + len(comp.candidate_set) == 0
    assert comp.xdim + comp.ydim == len(inputs) + len(outputs)
    for attribute in ['model_kwargs', 'misc_states', 'misc_costs', 'misc_coeff_train', 'misc_coeff_test',
                      'interpolator', 'training_data']:
        assert isinstance(getattr(comp, attribute), Serializable)

    # Test partial validation
    variables = [Variable(distribution=f'U(0, {np.random.rand()})') for i in range(10)]
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
            misc_states[alpha][beta] = LagrangeState(weights={'x': np.random.rand(3)}, x_grids={'x': np.random.rand(3)})

    c = Component(simple_model, variables[:5], outputs=variables[5:],
                  model_kwargs={'error': 0.2}, model_fidelity=(2,), data_fidelity=(3, 2), surrogate_fidelity=(3, 3),
                  active_set=active_set, misc_states=misc_states, misc_costs=misc_costs)
    serialize_kwargs = {'training_data': {'save_path': tmp_path / 'training_data.pkl'}}
    c2 = c.serialize(serialize_kwargs=serialize_kwargs)
    c3 = Component.deserialize(c2)
    assert c3 == c


class CustomKwargs(StringKwargs):
    def __init__(self, number, **kwargs):
        self.number = number
        super().__init__(**kwargs)

    def __str__(self):
        return f'CustomKwargs({", ".join([str(self.number)] + [f"{key}={value}" for key, value in self.data.items()])})'


@dataclass
class CustomInterpolator(Interpolator, Base64Serializable):
    kernel_width: float = 0.1
    kernel_type: str = 'gaussian'

    def refine(self, *args):
        pass

    def predict(self, *args):
        pass

    def gradient(self, *args):
        pass

    def hessian(self, *args):
        pass


@dataclass
class CustomInterpolatorState(InterpolatorState, StringSerializable):
    knots: list[ArrayLike, ...] = field(default_factory=list)


@dataclass
class CustomDataStorage(TrainingData, StringSerializable):
    length: int = 40
    width: int = 2

    def get(self, *args):
        pass

    def set(self, *args):
        pass

    def refine(self, *args):
        pass

    def set_errors(self, *args):
        pass

    def impute_missing_data(self, *args):
        pass

    def clear(self):
        pass


def test_custom_data_classes():
    # Test custom args/kwargs, states, interpolator, etc.
    c = Component(simple_model, Variable(), Variable(), interpolator=CustomInterpolator(), data_fidelity=(2,),
                  training_data=CustomDataStorage(), model_kwargs=CustomKwargs(100, hello=2),
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
            misc_states[alpha][beta] = LagrangeState(weights={'x': np.random.rand(5)}, x_grids={'x': np.random.rand(5)})
    c = Component(simple_model, [Variable() for i in range(3)], [Variable() for i in range(3)],
                  active_set=active_set, misc_costs=misc_costs, misc_states=misc_states,
                  model_kwargs={'output_path': '.', 'opts': {'max_iter': 1000}}, model_fidelity=(3,))

    savepath = tmp_path / 'component.yml'
    YamlLoader.dump(c, savepath)
    c_load = YamlLoader.load(savepath)
    assert c_load == c


@dataclass
class Extra:
    x1: float


def special_model(x, name='hello', frac=0.2, model_fidelity=(1,), output_path='.', output_vars=None):
    t1 = time.time()
    ret_dict = {'y1': x['x1'] + x['x2'], 'y2': x['x2'] * x['x3']}
    if np.random.rand() < frac:
        raise ValueError(f'Random error: {uuid.uuid4().hex}')
    mult = 1 if name == 'hello' else 2.5
    err = mult * (1 / model_fidelity[0])
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

    inputs = [Variable('x1'), Variable('x2'), Variable('x3')]
    outputs = [Variable('y1', nominal=0.1), Variable('y2'), Variable('y3')]
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
        comp = Component(special_model, inputs, outputs, model_fidelity=(2,))
        shape = (5, 2, 2)
        case = {'x1': np.random.rand(*shape), 'x2': np.random.rand(*shape), 'x3': np.random.rand(*shape)}
        ret = comp.call_model(case, model_fidelity='best', output_path=tmp_path, executor=executor)
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


def test_misc_coeff():
    """Test the iterative calculation of MISC coefficients"""
    def model(x, model_fidelity=(0, 0)):
        y = x
        return y
    comp = Component(model, model_fidelity=(2, 3), data_fidelity=(2, 1), surrogate_fidelity=(1,))
    max_ind = comp.model_fidelity + comp.max_beta

    for idx in itertools.product(*[range(m) for m in max_ind]):
        # Activate the next index
        alpha, beta = idx[:len(comp.model_fidelity)], idx[len(comp.model_fidelity):]
        neighbors = comp._neighbors(alpha, beta, forward=True)
        s = {(alpha, beta)}
        comp.update_misc_coeff(IndexSet(s), index_set='train')
        if (alpha, beta) in comp.candidate_set:
            comp.candidate_set.remove((alpha, beta))
        else:
            # Only for initial index which didn't come from the candidate set
            comp.update_misc_coeff(IndexSet(s), index_set='test')
        comp.active_set.update(s)

        comp.update_misc_coeff(neighbors, index_set='test')
        comp.candidate_set.update(neighbors)

        # Check all data structures are consistent
        coeff_sum = 0
        for a, b, coeff in comp.misc_coeff_train:
            assert (a, b) in comp.active_set and (a, b) not in comp.candidate_set
            coeff_sum += coeff
        assert coeff_sum == 1

        coeff_sum = 0
        for a, b, coeff in comp.misc_coeff_test:
            assert (a, b) in comp.active_set.union(comp.candidate_set)
            coeff_sum += coeff
        assert coeff_sum == 1


def test_sparse_grid(plots=False):
    """Simple cos test from Jakeman (2022)"""
    def model(inputs, model_fidelity=(0,)):
        alpha = np.atleast_1d(model_fidelity)
        eps = (1/5) * 2.0**(-alpha[..., 0])
        y = np.cos(np.pi/2 * (inputs['x'] + 4/5 + eps))
        return {'y': y}

    # Construct MISC surrogate from an index set
    Ik = [((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((2,), (0,)), ((1,), (1,)), ((0,), (2,)), ((1,), (2,)),
          ((2,), (1,)), ((2,), (2,))]
    x = Variable(distribution='U(-1, 1)')
    y = Variable()
    truth_alpha = (15,)
    comp = Component(model, x, y, model_fidelity=(2,), data_fidelity=(2,), vectorized=True)
    for alpha, beta in Ik:
        comp.activate_index(alpha, beta)
    N = 100
    xg = np.linspace(-1, 1, N)
    yt = comp.predict({'x': xg}, use_model=truth_alpha)['y']
    y_surr = comp.predict({'x': xg})['y']
    l2_error = relative_error(y_surr, yt)
    assert l2_error < 0.1

    # Plot results for each fidelity of the MISC surrogate
    if plots:
        fig, axs = plt.subplots(3, 3, sharey='row', sharex='col')
        for alpha in range(3):
            for beta in range(3):
                ax = axs[2-alpha, beta]
                xi, yi = comp.training_data.get((alpha,), (beta,))
                y_interp = comp.interpolator.predict({'x': xg}, comp.misc_states[(alpha,), (beta,)], (xi, yi))
                s = rf'$\hat{{f}}_{{{alpha}, {beta}}}$'
                ax.plot(xg, y_interp['y'], '--k', label=r'{}'.format(s), linewidth=1.5)
                s = rf'$\hat{{f}}_{alpha}$'
                ax.plot(xg, model({'x': xg}, (alpha,))['y'], '--b', label=r'{}'.format(s), linewidth=2)
                ax.plot(xg, yt, '-r', label=r'$f$', linewidth=2)
                ax.plot(xi['x'], yi['y'], 'or')
                ax_default(ax, r'$x$' if alpha == 0 else '', r'$f(x)$' if beta == 0 else '', legend=True)

        fig.text(0.5, 0.02, r'Increasing surrogate fidelity ($\beta$) $\rightarrow$', ha='center', fontweight='bold')
        fig.text(0.02, 0.5, r'Increasing model fidelity ($\alpha$) $\rightarrow$', va='center', fontweight='bold',
                 rotation='vertical')
        fig.set_size_inches(3 * 3, 3 * 3)
        fig.tight_layout(pad=3, w_pad=1, h_pad=1)
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(xg, yt, '-r', linewidth=2, label='Model')
        ax.plot(xg, y_surr, '--k', linewidth=1.5, label='MISC surrogate')
        ax_default(ax, r'$x$', r'$f(x)$', legend=True)
        plt.show()


def test_polynomial_regression(plots=False):
    """Test refining data and surrogate fidelity using polynomial regression."""
    # Use x^3 on (0, 1)
    def model(inputs):
        return {'y': inputs['x'] ** 3}

    x = Variable(distribution='U(0, 1)')
    comp = Component(model, x, 'y', data_fidelity=(2,), surrogate_fidelity=(2,), vectorized=True,
                     interpolator=Linear(regressor='RidgeCV', regressor_opts={'alphas': np.logspace(-3, 3, 10)}))
    Ik = [((), (0, 0)), ((), (0, 1)), ((), (1, 0)), ((), (2, 0)), ((), (1, 1)), ((), (0, 2)), ((), (1, 2)),
          ((), (2, 1)), ((), (2, 2))]
    for alpha, beta in Ik:
        with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
            comp.activate_index(alpha, beta)
    xg = np.linspace(0, 1, 100)
    yt = comp.predict({'x': xg}, use_model='best')['y']
    ysurr = comp.predict({'x': xg})['y']
    assert relative_error(ysurr, yt) < 0.1

    # Plot results for each fidelity of the MISC surrogate
    if plots:
        fig, axs = plt.subplots(3, 3, sharey='row', sharex='col')
        for beta0 in range(3):
            for beta1 in range(3):
                ax = axs[2 - beta1, beta0]
                xi, yi = comp.training_data.get((), (beta0, beta1))
                y_interp = comp.interpolator.predict({'x': xg}, comp.misc_states[(), (beta0, beta1)], (xi, yi))
                s = rf'$\hat{{f}}_{{{beta0}, {beta1}}}(x)$'
                ax.plot(xg, y_interp['y'], '--k', label=r'{}'.format(s), linewidth=1.5)
                ax.plot(xg, yt, '-r', label=r'$f(x)$', linewidth=2)
                ax.plot(xi['x'], yi['y'], 'or')
                ax_default(ax, r'$x$' if beta0 == 0 else '', r'$f(x)$' if beta1 == 0 else '', legend=True)

        fig.text(0.5, 0.02, r'Increasing data fidelity ($\beta_0$) $\rightarrow$', ha='center', fontweight='bold')
        fig.text(0.02, 0.5, r'Increasing surrogate fidelity ($\beta_1$) $\rightarrow$', va='center', fontweight='bold',
                 rotation='vertical')
        fig.set_size_inches(3 * 3, 3 * 3)
        fig.tight_layout(pad=3, w_pad=1, h_pad=1)
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(xg, yt, '-r', linewidth=2, label='Model')
        ax.plot(xg, ysurr, '--k', linewidth=1.5, label='MISC surrogate')
        ax_default(ax, r'$x$', r'$f(x)$', legend=True)
        plt.show()


def test_field_quantity():
    """Test approximation of a field quantity using MISC. Both input and output."""
    def model(inputs, pressure_coords=None):
        if pressure_coords is None:
            pressure_coords = np.linspace(-1, 1, 200)
        delta = np.atleast_1d(inputs['delta'])                      # Scalar ~ U(0, 1)
        pressure_x = np.atleast_1d(inputs['pressure_x'])            # Field (N,)
        pressure_y = np.atleast_1d(inputs['pressure_y'])            # Field (N,)
        vel_x = delta[..., np.newaxis] * np.cos(pressure_coords)    # Field (N,)
        vel_y = delta[..., np.newaxis] * np.sin(pressure_coords)    # Field (N,)
        amp = delta * np.mean(pressure_x + pressure_y, axis=-1)     # Scalar

        return {'amp': amp, 'vel_x': vel_x, 'vel_y': vel_y}

    dof = 200
    num_samples = 50
    delta = Variable(distribution='U(0, 1)')
    gamma = Variable(distribution='U(1, 2)')

    # Compute compression maps from field quantity data
    grid = np.linspace(-1, 1, dof)
    delta_samples = delta.sample(num_samples)
    gamma_samples = gamma.sample(num_samples)
    pressure_x = delta_samples[..., np.newaxis] * np.sin(grid)
    pressure_y = gamma_samples[..., np.newaxis] * np.cos(grid)
    pressure_matrix = {'pressure_x': pressure_x, 'pressure_y': pressure_y}
    pressure = Variable(compression=SVD(rank=2, data_matrix=pressure_matrix, coords=grid,
                                        fields=['pressure_x', 'pressure_y']), name='pressure')
    ret = model({'delta': delta_samples, 'pressure_x': pressure_x, 'pressure_y': pressure_y})
    vel_x, vel_y = ret['vel_x'], ret['vel_y']
    vel_matrix = {'vel_x': vel_x, 'vel_y': vel_y}
    vel = Variable(compression=SVD(rank=4, data_matrix=vel_matrix, coords=grid, fields=['vel_x', 'vel_y']))

    amp = Variable()

    max_beta = (3, 3)
    comp = Component(model, [delta, pressure], [amp, vel], data_fidelity=max_beta, vectorized=True)
    for idx in itertools.product(*[range(beta+1) for beta in max_beta]):
        comp.activate_index((), idx)

    xtest = {'delta': delta.sample(10)}
    pressure_samples = pressure.sample_domain(10)  # (10, rank)
    xtest.update({f'pressure{LATENT_STR_ID}{i}': pressure_samples[:, i] for i in range(pressure_samples.shape[1])})
    xtest_model = to_model_dataset(xtest, comp.inputs)[0]

    ysurr = comp.predict(xtest)
    ysurr = to_model_dataset(ysurr, comp.outputs)[0]
    ymodel = comp.predict(xtest_model, use_model='best', pressure_coords=grid)
    errors = {var: relative_error(ysurr[var], ymodel[var]) for var in ymodel}
    assert all([errors[var] < 0.01 for var in errors])


def test_comp_jacobian_and_hessian():
    f1 = lambda theta: 2 * theta['x1'] ** 2 * theta['x2'] + np.cos(theta['x3'])
    f2 = lambda theta: np.exp(theta['x2']) * theta['x1'] + np.sin(theta['x3']) * theta['x2']
    fun = lambda theta: {'y1': f1(theta), 'y2': f2(theta)}

    x1, x2, x3 = (Variable('x1', distribution='U(0, 2)'), Variable('x2', distribution='U(-1, 1)'),
                  Variable('x3', distribution='U(-3.14, 3.14)'))
    y1, y2 = Variable('y1'), Variable('y2')
    max_beta = (4, 4, 5)

    surr = Component(fun, [x1, x2, x3], [y1, y2], data_fidelity=max_beta, vectorized=True)
    for idx in itertools.product(*[range(beta) for beta in max_beta]):
        surr.activate_index((), idx)

    N = (5, 6)

    def fun_vec(theta: np.ndarray):
        theta_dict = {var: theta[..., i] for i, var in enumerate(['x1', 'x2', 'x3'])}
        y_dict = fun(theta_dict)
        y_vec = np.concatenate([y_dict[var][..., np.newaxis] for var in ['y1', 'y2']], axis=-1)
        return y_vec

    xtest = {'x1': x1.sample(N), 'x2': x2.sample(N), 'x3': x3.sample(N)}
    xvec = np.concatenate([xtest[var][..., np.newaxis] for var in ['x1', 'x2', 'x3']], axis=-1)

    jac_truth = approx_jac(fun_vec, xvec)
    jac_surr = surr.gradient(xtest, derivative='first')
    jac_surr_vec = np.concatenate([np.expand_dims(jac_surr[var], axis=-2) for var in ['y1', 'y2']], axis=-2)
    assert np.allclose(jac_truth, jac_surr_vec, rtol=1e-2, atol=1e-2)

    hess_truth = approx_hess(fun_vec, xvec)
    hess_surr = surr.hessian(xtest)
    hess_surr_vec = np.concatenate([np.expand_dims(hess_surr[var], axis=-3) for var in ['y1', 'y2']], axis=-3)
    assert np.allclose(hess_truth, hess_surr_vec, rtol=1e-1, atol=1e-1)


def test_get_training_data():
    def model(inputs, model_fidelity=(0,)):
        return {'y': inputs['x'] ** 2 + model_fidelity[0]}

    x = Variable(domain=(0, 3))
    y = Variable()
    comp = Component(model, x, y, model_fidelity=(3,), data_fidelity=(3,))
    for idx in itertools.product(*[range(alpha+1) for alpha in comp.model_fidelity],
                                 *[range(beta+1) for beta in comp.data_fidelity]):
        a = idx[:len(comp.model_fidelity)]
        b = idx[len(comp.model_fidelity):]
        comp.activate_index(a, b)

    xtrain, ytrain = comp.get_training_data()
    xtrue, ytrue = comp.training_data.get(comp.model_fidelity, comp.max_beta)
    assert all([np.allclose(xtrain[var], xtrue[var]) for var in xtrue])
    assert all([np.allclose(ytrain[var], ytrue[var]) for var in ytrue])


def test_compression_empty_fields():
    """Make sure compression skips samples that are nan or None."""
    A = Variable('A', distribution='U(-5, 5)')
    p = Variable('p')
    def model(inputs, pct_fail=0):
        if np.random.rand() < pct_fail:
            raise ValueError('Sorry this model just fails sometimes')

        coords = np.linspace(-2, 2, 100)
        amp = np.atleast_1d(inputs['A'])
        res = amp[..., np.newaxis] * np.tanh(coords)
        if len(amp.shape) == 1 and amp.shape[0] == 1:
            res = np.squeeze(res, axis=0)
        return {'p': res, 'p_coords': coords, 'y': amp ** 2}

    # Generate SVD data matrix
    rank = 4
    samples = A.sample(50)
    outputs = model({str(A): samples})
    data_matrix = outputs[p].T  # (dof, num_samples)
    p.compression = SVD(rank=rank, coords=outputs['p_coords'], data_matrix=data_matrix)

    comp = Component(model, [A], [p, 'y'], pct_fail=0.3)
    samples = A.sample(50)
    outputs = comp.call_model({'A': samples})
    compressed, y_vars = to_surrogate_dataset(outputs, comp.outputs, del_fields=True)
    recon, _ = to_model_dataset(compressed, comp.outputs, del_latent=True)

    indices = list(outputs.get('errors', {}).keys())
    error_cases = np.full(50, False)
    error_cases[indices] = True

    assert all([np.all(np.isnan(compressed[var][error_cases])) for var in y_vars])
    assert all([np.all(~np.isnan(compressed[var][~error_cases])) for var in y_vars])

    assert all([np.all(np.isnan(recon[var][error_cases, ...])) for var in comp.outputs])
    assert all([np.all(~np.isnan(recon[var][~error_cases, ...])) for var in comp.outputs])

    for original, reconstruct in zip(outputs['p'][~error_cases], recon['p'][~error_cases]):
        assert relative_error(reconstruct, original) < 1e-6
