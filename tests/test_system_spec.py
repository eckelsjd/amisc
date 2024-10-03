"""Test that the [`SystemSurrogate`][amisc.system.SystemSurrogate] class validation works."""
import concurrent.futures
import os
import shutil
from pathlib import Path

from amisc import Variable, Component, System, YamlLoader


def simple_model(x, output_path='.'):
    return x


def test_basic_init(tmp_path):
    """Test inputs/output/graph and root directory/logger/executor initialization."""
    comp1 = Component(simple_model, [Variable(l) for l in 'abc'], [Variable(l) for l in 'def'], name='comp1')
    comp2 = Component(simple_model, [Variable(l) for l in 'dghi'], [Variable(l) for l in 'jkl'], name='comp2')
    comp3 = Component(simple_model, [Variable(l) for l in 'lmno'], [Variable(l) for l in 'pqra'], name='comp3')
    system = System(comp1, [comp2], comp3)
    assert list(system.graph.edges()) == [('comp1', 'comp2'), ('comp2', 'comp3'), ('comp3', 'comp1')]
    assert system.inputs.keys() == {'b', 'c', 'g', 'h', 'i', 'm', 'n', 'o'}
    assert system.outputs.keys() == {'d', 'e', 'f', 'j', 'k', 'l', 'p', 'q', 'r', 'a'}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        paths = [tmp_path, tmp_path / 'amisc_test']
        for p in paths:
            s1 = System(comp1, comp2, comp3, root_dir=p, executor=executor)
            assert s1.root_dir.name.startswith('amisc_')
            assert (s1.root_dir / 'surrogates').is_dir()
            for comp in s1.components:
                assert (s1.root_dir / 'components' / comp.name).is_dir()
                assert comp.executor == s1.executor
                assert comp.logger.handlers[0].baseFilename == s1.logger.handlers[0].baseFilename

    serialize_kwargs = {comp.name: {'training_data': {'save_path': tmp_path / f'{comp.name}_training_data.pkl'}}
                        for comp in s1.components}
    assert System.deserialize(s1.serialize(serialize_kwargs=serialize_kwargs)) == s1


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
