"""Test variables and variable lists. Including tests for compression, normalization, and distributions."""
import numpy as np
import pytest

from amisc import YamlLoader
from amisc.compression import SVD
from amisc.utils import relative_error
from amisc.variable import Variable, VariableList


def test_load_and_dump_variables(tmp_path):
    """Make sure yaml serialization works for variables."""
    variables = [Variable('a', description='Altitude', units='m', distribution='U(0, 1)'),
                 Variable(u'Φ', description='Base width', units='m', distribution='N(0, 1)'),
                 Variable('p', description='Pressure', units='Pa', domain=(1e6, 2e6),
                          compression={'method': 'svd', 'interpolate_opts': {'kernel': 'gaussian', 'neighbors': 10},
                                       'rank': 10, 'energy_tol': 0.95})
                 ]
    YamlLoader.dump(variables, tmp_path / 'variables.yml')
    variables_load = YamlLoader.load(tmp_path / 'variables.yml')

    for v in variables_load:
        v_old = variables[variables.index(v)]
        assert v_old.get_domain() == v.get_domain()
        assert v_old.description == v.description
        assert v_old.compression == v.compression

    x = Variable()
    y1, y2 = [Variable(), Variable()]
    x_vars = [Variable(distribution='U(0, 1)') for i in range(5)]
    assert x.name == 'x'
    assert y1.name.startswith('X_')
    assert y2.name.startswith('X_')
    assert all([x_vars[i].name.startswith('X_') for i in range(5)])


def test_single_norm():
    norms = ['linear(0.5, 2)', 'log(10)', 'minmax(lb_norm=-1, ub_norm=1)', 'zscore(2, 1)']
    dists = ['U(0.1, 2)', 'N(3, 0.5)']

    for norm_method in norms:
        for dist_method in dists:
            variable = Variable(norm=norm_method, distribution=dist_method)
            x = variable.sample_domain(10)
            xnorm = variable.normalize(x)
            xtilde = variable.normalize(xnorm, denorm=True)
            assert np.allclose(x, xtilde), (f'Failed single normalization test for '
                                            f'distribution: {dist_method}, norm: {norm_method}')


def test_two_norms():
    cases = [['log(offset=1)', 'minmax'], ['linear(0.25, 5)', 'minmax'], ['log', 'linear(-2, 20)'],
             ['zscore(3, 0.5)', 'minmax']]
    dists = ['Uniform(1e-8, 1e-5)', 'Normal(3, 0.5)']

    for dist_method in dists:
        for case in cases:
            var1 = Variable(distribution=dist_method, norm=case)
            var2 = Variable(distribution=dist_method, norm=list(reversed(case)))
            x = var1.sample_domain(10)
            xnorm1 = var1.normalize(x)
            xnorm2 = var2.normalize(x)
            xtilde1 = var1.normalize(xnorm1, denorm=True)
            xtilde2 = var2.normalize(xnorm2, denorm=True)
            assert np.allclose(x, xtilde1), (f'Failed the two-normalization test for distribution: {dist_method}, '
                                             f'norm: {case}')
            assert np.allclose(x, xtilde2), (f'Failed the two-normalization test for distribution: '
                                             f'{dist_method}, norm: {var2.norm}')


def test_many_norms():
    cases = [['linear(0.1)', 'zscore(mu=-1, std=0.25)', 'log(base=10, offset=5)', 'minmax(lb_norm=-1, ub_norm=1)'],
             ['log10(offset=11)', 'minmax', 'linear(-2, 1e-6)']]
    dist = 'U(-10, -5)'
    for case in cases:
        variable = Variable(distribution=dist, norm=case)
        x = variable.sample_domain(50)
        xnorm = variable.normalize(x)
        xtilde = variable.normalize(xnorm, denorm=True)
        assert np.allclose(x, xtilde), f'Failed the many-normalization test for distribution: {dist}, norm: {case}'


def test_nominal_and_domain():
    variable = Variable(distribution='U(1, 10)', norm='log10')
    nominal = variable.get_nominal()
    bds = variable.get_domain()
    assert nominal == (10 + 1)/2
    assert bds == (1, 10)

    variable = Variable(distribution='N(5, 1)', norm='minmax')
    nominal= variable.get_nominal()
    bds = variable.get_domain()
    assert nominal == 5
    assert bds == (2, 8)


def test_compression_1d():
    """Test svd compression for a single-input, single-output, 1d field quantity."""
    A = Variable('A', distribution='U(-5, 5)')
    p = Variable('p')
    def model(inputs, coords):
        amp = inputs['A']
        return {'p': amp[..., np.newaxis] * np.tanh(np.squeeze(coords))}

    # Generate SVD data matrix
    rank = 4
    samples = A.sample(50)
    svd_coords = np.linspace(-2, 2, 200)
    outputs = model({str(A): samples}, svd_coords)
    data_matrix = outputs[p].T  # (dof, num_samples)
    p.compression = SVD(rank=rank, coords=svd_coords, data_matrix=data_matrix)

    # Test compression
    coarse_shape = 25
    coarse_coords = np.linspace(-2, 2, coarse_shape)
    num_test = (5, 20)
    samples = A.sample(num_test)
    outputs = model({str(A): samples}, coarse_coords)
    outputs_reduced = p.compress(outputs, coords=coarse_coords)
    outputs_reconstruct = p.reconstruct(outputs_reduced)

    y = outputs[p]
    yhat = outputs_reconstruct[p]
    assert relative_error(yhat, y) < 0.01

    # Test sampling latent coefficients
    latent_min = np.min(outputs_reduced['latent'], axis=(0, 1))
    latent_max = np.max(outputs_reduced['latent'], axis=(0, 1))
    p.domain = (-1, 1)
    assert len(p.get_domain()) == rank  # inferred from compression rank
    p.domain = [(lmin, lmax) for lmin, lmax in zip(latent_min, latent_max)]
    p.norm = 'linear(0.5)'
    domain = p.get_domain()
    samples = p.sample_domain(num_test)
    nominal = p.sample(num_test)
    lb = np.atleast_1d([d[0] for d in domain])
    ub = np.atleast_1d([d[1] for d in domain])
    assert len(domain) == rank
    assert samples.shape == num_test + (rank,)
    assert nominal.shape == num_test + (rank,)
    assert np.all(np.logical_and(lb < samples, samples < ub))
    assert np.allclose(nominal - (lb+ub)/2, 0)


def test_compression_fields():
    """Test compression for a multi-field quantity."""
    num_grid = 200
    num_samples = 50
    delta = Variable(distribution='U(0, 1)')
    gamma = Variable(distribution='U(1, 2)')
    grid = np.linspace(-1, 1, num_grid)
    delta_samples = delta.sample(num_samples)
    gamma_samples = gamma.sample(num_samples)
    pressure_x = delta_samples[..., np.newaxis] * np.sin(grid)
    pressure_y = gamma_samples[..., np.newaxis] * np.cos(grid)
    pressure_matrix = np.concatenate((pressure_x[..., np.newaxis], pressure_y[..., np.newaxis]),
                                     axis=-1).reshape((num_samples, -1))
    pressure = Variable(compression=SVD(reconstruction_tol=0.001, data_matrix=pressure_matrix.T, coords=grid,
                                        fields=['pressure_x', 'pressure_y']), name='pressure')

    latent = pressure.compress({'pressure_x': pressure_x, 'pressure_y': pressure_y})
    pressure_reconstruct = pressure.reconstruct(latent)
    assert relative_error(pressure_reconstruct['pressure_x'], pressure_x) < 0.01
    assert relative_error(pressure_reconstruct['pressure_y'], pressure_y) < 0.01

    coarse_grid = 20
    coarse_coords = np.linspace(-1, 1, coarse_grid)
    num_test = (5, 10)
    delta_samples = delta.sample(num_test)
    gamma_samples = gamma.sample(num_test)
    pressure_x = delta_samples[..., np.newaxis] * np.sin(coarse_coords)
    pressure_y = gamma_samples[..., np.newaxis] * np.cos(coarse_coords)

    latent = pressure.compress({'pressure_x': pressure_x, 'pressure_y': pressure_y}, coords=coarse_coords)
    pressure_reconstruct = pressure.reconstruct(latent)

    assert relative_error(pressure_reconstruct['pressure_x'], pressure_x) < 0.01
    assert relative_error(pressure_reconstruct['pressure_y'], pressure_y) < 0.01


def test_compression_coords():
    """Test compression with different sets of coordinates in a single call."""
    A = Variable('A', distribution='U(-5, 5)')
    p = Variable('p')

    def model(inputs, coords):
        amp = inputs['A']
        return {'p': amp[..., np.newaxis] * np.tanh(np.squeeze(coords))}

    # Generate SVD data matrix
    rank = 4
    samples = A.sample(50)
    svd_coords = np.linspace(-2, 2, 200)
    outputs = model({str(A): samples}, svd_coords)
    data_matrix = outputs[p].T  # (dof, num_samples)
    p.compression = SVD(rank=rank, coords=svd_coords, data_matrix=data_matrix)

    # Test compression
    coarse_shapes = [25, 50, 75, 100, 200]
    coarse_coords = np.array([np.linspace(-2, 2, s) for s in coarse_shapes], dtype=object)
    num_test = (5, 20)
    samples = A.sample(num_test)
    outputs = [model({str(A): samples}, coords) for coords in coarse_coords]
    outputs_obj = {var: np.empty(len(coarse_shapes), dtype=object) for var in outputs[0]}
    for var in outputs[0]:
        for i, out in enumerate(outputs):
            outputs_obj[var][i] = out[var]
    outputs_reduced = p.compress(outputs_obj, coords=coarse_coords)
    outputs_reconstruct = p.reconstruct(outputs_reduced)

    y = outputs_obj[p]
    yhat = outputs_reconstruct[p]
    assert all([relative_error(yhat[i], y[i]) < 0.01 for i in range(len(coarse_shapes))])


def test_compression_interpolation():
    """Test to_grid and from_grid interpolation for compressed data."""
    # Generate compression data
    num_features = 50
    svd_coords = np.linspace(-1, 1, num_features)
    svd = SVD(coords=svd_coords, fields=['r'])

    # Test object arrays to/from
    coarse_shapes = [5, 10, 25, 47]
    test_coords = np.empty(len(coarse_shapes), dtype=object)
    test_values = {'r': np.empty(len(coarse_shapes), dtype=object)}
    for i in range(len(coarse_shapes)):
        test_coords[i] = np.linspace(-1, 1, coarse_shapes[i])
        test_values['r'][i] = np.tanh(test_coords[i])

    on_grid = svd.interpolate_to_grid(test_coords, test_values)
    from_grid = svd.interpolate_from_grid(on_grid, test_coords)
    assert on_grid.shape == (len(coarse_shapes), num_features)
    assert all([relative_error(from_grid['r'][i], test_values['r'][i]) < 0.001 for i in range(len(coarse_shapes))])

    # Test object arrays with grid coords
    test_values = {'r': np.empty(len(coarse_shapes), dtype=object)}
    for i in range(len(coarse_shapes)):
        test_values['r'][i] = on_grid[i]
    on_grid = svd.interpolate_to_grid(svd_coords, test_values)
    from_grid = svd.interpolate_from_grid(on_grid, svd_coords)
    assert on_grid.shape == (len(coarse_shapes), num_features)
    assert np.allclose(on_grid, from_grid['r'])

    # Test vectorized arrays to/from
    vec_coords = np.empty((10, 5), dtype=object)
    vec_values = np.empty((10, 5, 27))
    for i in np.ndindex(vec_coords.shape):
        vec_coords[i] = np.sort(np.random.uniform(-1, 1, 27))
        vec_values[i] = np.tanh(vec_coords[i])
    on_grid = svd.interpolate_to_grid(vec_coords, {'r': vec_values})
    from_grid = svd.interpolate_from_grid(on_grid, vec_coords)
    assert on_grid.shape == (10, 5, num_features)
    assert all([relative_error(from_grid['r'][i], vec_values[i]) < 0.001 for i in np.ndindex(vec_coords.shape)])

    # Test vectorized arrays with object grid coord
    test_coords = np.empty((10, 5), dtype=object)
    for i in np.ndindex(test_coords.shape):
        test_coords[i] = np.linspace(-1, 1, np.random.randint(20, 50))
    from_grid = svd.interpolate_from_grid(on_grid, test_coords)
    for i in np.ndindex(test_coords.shape):
        to_grid = svd.interpolate_to_grid(test_coords[i], {'r': from_grid['r'][i][np.newaxis, ...]})
        assert relative_error(to_grid[0], on_grid[i]) < 0.001
        assert from_grid['r'][i].shape[0] == test_coords[i].shape[0]


@pytest.mark.skip(reason='Not implemented')
def test_compression_nd():
    """Test SVD compression for high-dimensional data."""
    # TODO
    pass


def test_uniform_dist():
    shape = (50,)
    true_lb = [-2, 1.1, 11.3]
    true_ub = [-1, 1.2, 94]
    for i in range(len(true_lb)):
        v = Variable(distribution=f'Uniform({true_lb[i]}, {true_ub[i]})')
        samples = v.sample(shape)
        pdf = v.pdf(samples)
        assert np.all(np.logical_and(true_lb[i] < samples, samples < true_ub[i]))
        assert np.allclose(pdf, 1 / (true_ub[i] - true_lb[i]))
        assert np.allclose(v.pdf(samples + true_ub[i]), 0)

    # Loguniform distribution
    v = Variable(distribution='LogUniform(1e-8, 1e-2)', norm='log10')
    norm_bds = v.normalize(v.get_domain())
    unnorm_bds = v.denormalize(norm_bds)
    samples = v.sample(shape)
    pdf = v.pdf(samples)
    samples_norm = v.normalize(samples)
    assert np.all(np.logical_and(norm_bds[0] < samples_norm, samples_norm < norm_bds[1]))
    assert np.all(np.logical_and(unnorm_bds[0] < samples, samples < unnorm_bds[1]))
    assert np.allclose(pdf, 1 / (samples * (np.log(unnorm_bds[1]) - np.log(unnorm_bds[0]))))


def test_normal_dist():
    shape = (10, 20, 50)
    true_means = [-5.2, 1, 2.7, 14.73, 100]
    true_stds = (np.random.rand(len(true_means)) * 0.1 + 0.05) * np.abs(true_means)
    variables = [Variable(distribution=f'Normal({true_means[i]}, {true_stds[i]})') for i in range(len(true_means))]
    for i, v in enumerate(variables):
        # Plain normal distribution
        samples = v.sample(shape)
        x = np.linspace(*v.get_domain(), 1000)
        pdf = v.pdf(x)
        assert relative_error(np.mean(samples), true_means[i]) < 0.05
        assert relative_error(x[np.argmax(pdf)], true_means[i]) < 0.05

    # Log normal distribution
    v = Variable(distribution='LogNormal(-2, 1, 2.718)', norm='log')
    samples = v.sample(shape)
    x = np.logspace(*v.normalize(v.get_domain()), 1000)
    pdf = v.pdf(x)
    mode = np.exp(-2 - 1**2)
    assert relative_error(np.mean(v.normalize(samples)), -2) < 0.05
    assert relative_error(x[np.argmax(pdf)], mode) < 0.05


def test_relative_and_tolerance_dist():
    shape = (10, 2)
    nominals = [1.1, -3, 1e6, 20.]
    dists = ['rel', 'tol']
    tols = [0.5, 1.1, 10]
    v = Variable()
    for nominal in nominals:
        for dist in dists:
            for tol in tols:
                v.nominal = nominal
                v.distribution = f'{dist}({tol})'
                samples = v.sample(shape)
                tol = tol if dist == 'tol' else (tol/100) * abs(nominal)
                bds = (nominal - tol, nominal + tol)
                assert np.all(np.logical_and(bds[0] < samples, samples < bds[1]))


def test_no_dist():
    shape = (10, 2)
    nominals = [-5.2, 1, 2.7, 14.73, 100]
    v = Variable(domain=(0, 1))
    for i, nominal in enumerate(nominals):
        v.nominal = nominal
        samples = v.sample(shape)
        x = np.linspace(*v.get_domain(), 100)
        pdf = v.pdf(x)
        assert np.allclose(samples, nominal)
        assert np.allclose(pdf, 1)


def test_variable_list(tmp_path):
    # Initialize several variables
    letters = 'abcdefgh'
    means = np.random.rand(len(letters)) * 10 - 5
    std = np.random.rand(len(letters)) * 3
    variables = [Variable(letter, tex=f'${letter}_{i}$', distribution=f'N({means[i]}, {std[i]})', norm='zscore')
                 for i, letter in enumerate(letters)]

    # Init from var, dict, list, or varlist
    l1 = VariableList(variables)
    l2 = VariableList({v: v for v in variables})
    l3 = VariableList(variables[0])
    for v in variables:
        l3.append(v)
    l4 = VariableList(l3)

    for i, v in enumerate(variables):
        assert l1[v] == v
        assert l2.get(v) == v
        assert l3[i] == v
        assert l4[v.name] == v.name

    # Fancy indexing
    indices = [(0, 3, 5), slice(1, 4), ['a', 'b', 'h', 'g', 'f']]
    for index in indices:
        subset1 = l1[index]
        for i, ele in enumerate(subset1):
            assert ele == l2[index][i]
            assert ele == l3[index][i]
            assert ele == l4[index][i]

    # Set/get bad values
    bad_keys = [2.3, np.sum, slice(None), 'a']
    bad_vals = [variables[0], variables[1], variables[2], 5.4]
    for i, key in enumerate(bad_keys):
        try:
            l1[key] = bad_vals[i]
            assert False
        except Exception:
            pass

    bad_keys = ['z', 'y', 'X2', 1.2, np.mean]
    for key in bad_keys:
        try:
            print(l4[key])
            assert False
        except Exception:
            pass

    # Delete and check
    del_idx = (0, -1, 'c')
    l1[-1] = Variable()
    assert l2[-1] != l1[-1]
    del l1[del_idx]
    assert l1[0] == 'b'
    assert l1[-1] == l2[-2]
    assert len(l1) == len(letters) - len(del_idx)

    # Load/dump from yaml
    YamlLoader.dump(l4, tmp_path / 'variable_list.yml')
    variables_load = YamlLoader.load(tmp_path / 'variable_list.yml')

    for v in variables_load:
        v_old = l4[l4.index(v)]
        assert v_old.get_domain() == v.get_domain()
        assert v_old.get_tex(units=True) == v.get_tex(units=True)
