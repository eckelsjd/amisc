from amisc.utils import get_logger, parse_function_string


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
