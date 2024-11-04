"""Test the package utilities."""
# ruff: noqa: E741
import time

import numpy as np
from scipy.linalg import lapack

from amisc.compression import SVD
from amisc.typing import LATENT_STR_ID
from amisc.utils import (
    _inspect_function,
    constrained_lls,
    format_inputs,
    get_logger,
    parse_function_string,
    to_model_dataset,
    to_surrogate_dataset,
)
from amisc.variable import Variable, VariableList


def test_inspect_function():
    """Test inspection of a function."""
    def func1(a, b=2, **kwargs):
        y1 = a + b
        return y1

    def another_function(x, y, z):
        hello = 'world'
        _there_is = 'no spoon'
        nothing = x * y + z
        left = 'right'
        return hello, _there_is, nothing, left

    def sum_function(*args, **kwargs):
        return sum(args) + sum(kwargs.values())

    test_cases = [(func1, ['a'], ['y1']),
                  (another_function, ['x', 'y', 'z'], ['hello', '_there_is', 'nothing', 'left']),
                  (sum_function, [], [])
                  ]

    for func, expected_args, expected_outputs in test_cases:
        args, outputs = _inspect_function(func)
        assert args == expected_args
        assert outputs == expected_outputs


def test_format_inputs():
    """Test broadcasting of input arrays to the same leading dimensions"""
    cases = [{'x': np.random.rand(10, 2, 5), 'y': np.random.rand(1), 'z': np.random.rand(1, 20), 'z2': 1.0},
             {'a': np.random.rand(10, 1, 2), 'b': np.random.rand(1, 5, 1), 'c': np.random.rand(1, 1)},
             {'x1': 1.0, 'x2': [2.0], 'x3': np.random.rand(20, 1), 'x4': np.random.rand(1, 20, 1)},
             {'x': np.random.rand(15), 'y': np.random.rand(15, 5, 2), 'z': np.random.rand(15, 3, 5)},
             {'a': np.random.rand(10, 2), 'b': np.random.rand(5,), 'c': np.random.rand(6, 1)}]
    results = [(10,), (10, 5), (20,), (15,), ()]
    for case, result in zip(cases, results):
        formatted_inputs, loop_shape = format_inputs(case)
        assert loop_shape == result, f"Loop shape mismatch: {loop_shape} != {result}"
        for key, value in formatted_inputs.items():
            og_array = np.atleast_1d(case[key])
            N = np.prod(result)
            l = len(result)
            if l > 0:
                assert value.shape[0] == N, f"Loop shape mismatch for {key}: {value.shape[0]} != {N}"
                assert value.shape[1:] == og_array.shape[l:], (f"Other shape mismatch for {key}: "
                                                               f"{value.shape[1:]} != {og_array.shape[l:]}")
            else:
                assert value.shape == og_array.shape, f"Shape mismatch for {key}: {value.shape} != {og_array.shape}"


def test_parse_function_string():
    """Test parsing a function/class string"""
    cases = ["MyClass(1, 'False', True, alpha=(2,2), hello='goodbye', s='()\":\"true')",
             'myfunc(1.0, 1e6, "12", *, std="file.py")',
             "AnotherClass(42, 'test', beta=[1,2,3], gamma={'key': 'value'})",
             'anotherfunc(3.14, "hello", *, flag=True)',
             "ComplexClass(0, 'complex', delta=(1,2,3), epsilon={'nested': {'key': 'value'}})"
             ]
    results = [('MyClass', [1, 'False', True], {'alpha': (2, 2), 'hello': 'goodbye', 's': '()":"true'}),
               ('myfunc', [1.0, 1e6, '12'], {'std': 'file.py'}),
               ('AnotherClass', [42, 'test'], {'beta': [1, 2, 3], 'gamma': {'key': 'value'}}),
               ('anotherfunc', [3.14, 'hello'], {'flag': True}),
               ('ComplexClass', [0, 'complex'], {'delta': (1, 2, 3), 'epsilon': {'nested': {'key': 'value'}}})
               ]
    for case, result in zip(cases, results):
        name, args, kwargs = parse_function_string(case)
        assert name == result[0]
        assert args == result[1]
        assert kwargs == result[2]


def test_logging():
    """Test logging and plotting utils"""
    logger = get_logger('tester', stdout=True)
    logger.info('Testing logger...')


def test_lls():
    """Test constrained linear least squares routine against scipy lapack."""
    X = 100
    Y = 100
    M = 10
    N = 10
    P = 1
    tol = 1e-8

    A = np.random.rand(X, Y, M, N)
    b = np.random.rand(X, Y, M, 1)
    C = np.random.rand(X, Y, P, N)
    d = np.random.rand(X, Y, P, 1)

    # custom solver
    t1 = time.time()
    alpha = np.squeeze(constrained_lls(A, b, C, d), axis=-1)  # (*, N)
    t2 = time.time()

    # Built in scipy solver
    alpha2 = np.zeros((X, Y, N))
    t3 = time.time()
    for i in range(X):
        for j in range(Y):
            Ai = A[i, j, ...]
            bi = b[i, j, ...]
            Ci = C[i, j, ...]
            di = d[i, j, ...]
            ret = lapack.dgglse(Ai, Ci, bi, di)
            alpha2[i, j, :] = ret[3]
    t4 = time.time()

    # Results
    diff = alpha - alpha2
    assert np.max(np.abs(diff)) < tol
    print(f'Custom CLLS time: {t2-t1} s. Scipy time: {t4-t3} s.')


def test_dataset_conversion():
    """Test conversion of datasets to/from model and surrogate forms."""
    # Data matrix for field
    param = np.random.rand(50)
    svd_coords = np.linspace(0, 10, 100)
    ux = np.sin(svd_coords)[..., np.newaxis] * param
    uy = np.cos(svd_coords)[..., np.newaxis] * param
    A = np.concatenate((ux, uy), axis=0)
    compression = SVD(data_matrix=A, rank=4, fields=['ux', 'uy'], coords=svd_coords)
    latent = compression.compress(A.T)  # (num_samples, rank)
    domain = list(zip(np.min(latent, axis=0), np.max(latent, axis=0)))

    # Scalar, normalized, and field quantity variables
    scalar = Variable('scalar', distribution='U(0, 1)')
    norm = Variable('norm', distribution='LN(0, 1)', norm='log10')
    field = Variable('f', domain=domain, compression=compression)
    vlist = VariableList([scalar, norm, field])

    # Surrogate dataset
    size = (10, 3)
    s = field.sample_domain(size)
    surr_ds = {f'{field}{LATENT_STR_ID}{i}': s[..., i] for i in range(s.shape[-1])}
    surr_ds.update({var: var.normalize(var.sample(size)) for var in [scalar, norm]})

    # To model dataset (denormalize/reconstruct)
    model_ds, fc = to_model_dataset(surr_ds, vlist, del_latent=True, f_coords=svd_coords)
    assert np.allclose(model_ds['scalar'], surr_ds['scalar'])
    assert np.allclose(norm.normalize(model_ds['norm']), surr_ds['norm'])
    assert np.allclose(fc['f_coords'], svd_coords)
    assert all([var in model_ds for var in ['scalar', 'norm', 'ux', 'uy']])
    assert all([LATENT_STR_ID not in var for var in model_ds])
    assert all([model_ds[var].shape == (*size, 100) for var in ['ux', 'uy']])
    latent = field.compress({var: model_ds[var] for var in ['ux', 'uy']})['latent']
    assert all([np.allclose(latent[..., i], surr_ds[f'{field}{LATENT_STR_ID}{i}']) for i in range(latent.shape[-1])])

    # Back to surrogate dataset (normalize/compress)
    surr_ds2, surr_vars = to_surrogate_dataset(model_ds, vlist, del_fields=True, **fc)
    assert all([v in surr_vars for v in surr_ds.keys()])
    assert all([var not in surr_ds2 for var in ['ux', 'uy']])
    assert all([np.allclose(surr_ds[var], surr_ds2[var]) for var in surr_ds2])
