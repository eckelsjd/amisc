import numpy as np

from amisc.utils import get_logger, parse_function_string, format_inputs


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
