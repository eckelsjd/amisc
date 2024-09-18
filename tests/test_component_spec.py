import itertools

import numpy as np

from amisc.component import IndexSet, MiscTree, ModelArgs, ModelKwargs


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
    # data = {(j, 3): {(i,): LagrangeState(weights=[np.random.rand(3)], x_grids=[np.random.rand(3)]) for i in range(4)}
    #         for j in range(2)}
    # tree = MiscTree(data=data, state_serializer=LagrangeState)
    # serialized_tree = tree.serialize()
    # deserialized_tree = MiscTree.deserialize(serialized_tree)
    #
    # for alpha, beta, data in tree:
    #     assert deserialized_tree[alpha, beta] == data


def test_component_validation():
    # Test completely empty
    # Test partial validation
    pass


def test_custom_data_classes():
    # Test custom args/kwargs, states, interpolator, etc.
    pass


def test_serialization():
    # Test json/yaml serialization
    pass


# import numpy as np
# import yaml
#
# from amisc.component import Component, LagrangeState
# from amisc.variable import Variable
#
# variables = [Variable(dist=f'U(0, {np.random.rand()})') for i in range(10)]
#
# def model(x, alpha, error=0.1, output_path='.'):
#     return alpha * error + x ** 2
#
# alphas = [(), (0,), (1,)]
# betas = [(0, 0), (0., 1.1), (1, 0)]
#
# active_set = []
# misc_states = {}
# misc_costs = {}
# for alpha in alphas:
#     for beta in betas:
#         active_set.append((alpha, beta))
#         misc_costs.setdefault(alpha, dict())
#         misc_states.setdefault(alpha, dict())
#         misc_costs[alpha][beta] = 1.0 if np.random.rand() < 0.5 else -1.0
#         misc_states[alpha][beta] = LagrangeState(weights=[np.random.rand(3)], x_grids=[np.random.rand(3)])
#
#
# c = Component(model=model, inputs=variables[:5], outputs=variables[5:], model_args=(1,), model_kwargs={'error': 0.2},
#               max_alpha=(2,), max_beta=(3, 2), interpolator='lagrange', active_set=active_set,
#               misc_states=misc_states, misc_costs=misc_costs)
#
# with open('tests/file.yml', 'w') as fd:
#     yaml.dump(c, fd)
#
# with open('tests/file.yml', 'r') as fd:
#     comp_load = yaml.load(fd, Loader=yaml.Loader)
#
# print(comp_load)
