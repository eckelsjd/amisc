import h5py
import numpy as np
import yaml

from amisc.utils import relative_error
from amisc.variable import Variable


def test_load_and_dump_variables(tmp_path):
    variables = [Variable(var_id='a', description='Altitude', units='m', dist='U(0, 1)'),
                 Variable(var_id=u'Φ', description='Base width', units='m', dist='N(0, 1)'),
                 Variable(var_id='p', description='Pressure', units='Pa', domain=(1e6, 2e6),
                          field={'compress_method': 'SVD', 'interp_args': {'kernel': 'gaussian', 'neighbors': 10},
                                 'compress_args': {'rank': 10, 'energy_tol': 0.95}})
                 ]
    with open(tmp_path / 'variables.yml', 'w') as fd:
        yaml.dump(variables, fd, allow_unicode=True)

    with open(tmp_path / 'variables.yml', 'r') as fd:
        variables_load = yaml.load(fd, yaml.Loader)

    for v in variables_load:
        v_old = variables[variables.index(v)]
        assert v_old.get_domain() == v.get_domain()
        assert v_old.description == v.description


def test_single_norm():
    norms = ['linear(1/2, 2)', 'log(10)', 'minmax(lb_norm=-1, ub_norm=1)', 'zscore(2, 1)']
    dists = ['U(0.1, 2)', 'N(3, 0.5)']

    for norm_method in norms:
        for dist_method in dists:
            variable = Variable(norm=norm_method, dist=dist_method)
            x = variable.sample_domain(10)
            xnorm = variable.normalize(x)
            xtilde = variable.normalize(xnorm, denorm=True)
            assert np.allclose(x, xtilde), (f'Failed single normalization test for '
                                            f'dist: {dist_method}, norm: {norm_method}')


def test_two_norms():
    cases = [['log(offset=1)', 'minmax'], ['linear(0.25, 5)', 'minmax'], ['log', 'linear(-2, 20)'],
             ['zscore(3, 0.5)', 'minmax']]
    dists = ['Uniform(1e-8, 1e-5)', 'Normal(3, 0.5)']

    for dist_method in dists:
        for case in cases:
            var1 = Variable(dist=dist_method, norm=case)
            var2 = Variable(dist=dist_method, norm=list(reversed(case)))
            x = var1.sample_domain(10)
            xnorm1 = var1.normalize(x)
            xnorm2 = var2.normalize(x)
            xtilde1 = var1.normalize(xnorm1, denorm=True)
            xtilde2 = var2.normalize(xnorm2, denorm=True)
            assert np.allclose(x, xtilde1), f'Failed the two-normalization test for dist: {dist_method}, norm: {case}'
            assert np.allclose(x, xtilde2), (f'Failed the two-normalization test for dist: '
                                             f'{dist_method}, norm: {var2.norm}')


def test_many_norms():
    cases = [['linear(0.1)', 'zscore(mu=-1, std=0.25)', 'log(base=10, offset=5)', 'minmax(lb_norm=-1, ub_norm=1)'],
             ['log10(offset=11)', 'minmax', 'linear(-2, 1e-6)']]
    dist = 'U(-10, -5)'
    for case in cases:
        variable = Variable(dist=dist, norm=case)
        x = variable.sample_domain(50)
        xnorm = variable.normalize(x)
        xtilde = variable.normalize(xnorm, denorm=True)
        assert np.allclose(x, xtilde), f'Failed the many-normalization test for dist: {dist}, norm: {case}'


def test_nominal_and_domain():
    variable = Variable(dist='U(1, 10)', norm='log10')
    nominal, norm_nominal = variable.get_nominal(norm=False), variable.get_nominal(norm=True)
    bds, norm_bds = variable.get_domain(norm=False), variable.get_domain(norm=True)
    assert nominal == (10 + 1)/2 and norm_nominal == (0 + 1)/2
    assert bds == (1, 10) and norm_bds == (0, 1)

    variable = Variable(dist='N(5, 1)', norm='minmax')
    nominal, norm_nominal = variable.get_nominal(norm=False), variable.get_nominal(norm=True)
    bds, norm_bds = variable.get_domain(norm=False), variable.get_domain(norm=True)
    assert nominal == 5 and norm_nominal == 0.5
    assert bds == (2, 8) and norm_bds == (0, 1)


def test_compression_1d(tmp_path):
    """Test svd compression for a single-input, single-output, 1d field quantity."""
    svd_file = tmp_path / 'pressure_svd.h5'
    A = Variable(var_id='A', dist='U(-5, 5)')
    p = Variable(var_id='p', field={'compress_method': 'SVD', 'compress_datapath': str(svd_file.resolve()),
                                    'compress_args': {'energy_tol': 0.99}})
    def model(inputs, coords):
        amp = inputs['A']
        return {'p': {'coord': coords, 'p': amp[..., np.newaxis] * np.tanh(np.squeeze(coords))}}

    # Generate SVD data matrix (TODO: refactor to compress.py)
    samples = A.sample(50)
    svd_coords = np.linspace(-2, 2, 200)
    outputs = model({str(A): samples}, svd_coords)[p]
    data_matrix = outputs[p].T  # (dof, num_samples)
    nan_idx = np.any(np.isnan(data_matrix), axis=0)
    data_matrix = data_matrix[:, ~nan_idx]
    u, s, vt = np.linalg.svd(data_matrix)
    energy_frac = np.cumsum(s ** 2 / np.sum(s ** 2))
    if energy_tol := p.field['compress_args'].get('energy_tol'):
        idx = int(np.where(energy_frac >= energy_tol)[0][0])
        rank = idx + 1
    elif rank := p.field['compress_args'].get('rank'):
        energy_tol = energy_frac[rank - 1]
    else:
        raise ValueError(f'Must specify either energy_tol or rank for SVD compression for variable "{p}".')
    projection_matrix = u[:, :rank]  # (dof, rank)

    with h5py.File(svd_file, 'w') as fd:
        fd.create_dataset('projection_matrix', data=projection_matrix)
        fd.create_dataset('data_matrix', data=data_matrix)
        fd.create_dataset('coord', data=svd_coords[... , np.newaxis])
        fd.create_dataset('rank', data=rank)
        fd.create_dataset('energy_tol', data=energy_tol)

    # Test compression
    coarse_shape = 25
    coarse_coords = np.linspace(-2, 2, coarse_shape)
    num_test = (5, 20)
    samples = A.sample(num_test)
    outputs = model({str(A): samples}, coarse_coords)[p]
    outputs_reduced = p.compress(outputs)
    outputs_reconstruct = p.reconstruct(outputs_reduced)

    y = outputs[p]
    yhat = outputs_reconstruct[p]

    assert relative_error(yhat, y) < 0.01


def test_compression_nd():
    """Test SVD compression for high-dimensional data."""
    # TODO
    pass


def test_uniform_dist():
    shape = (50,)
    true_lb = [-2 , 1.1, 11.3]
    true_ub = [-1, 1.2, 94]
    for i in range(len(true_lb)):
        v = Variable(dist=f'Uniform({true_lb[i]}, {true_ub[i]})')
        samples = v.sample(shape)
        pdf = v.pdf(samples)
        assert np.all(np.logical_and(true_lb[i] < samples, samples < true_ub[i]))
        assert np.allclose(pdf, 1 / (true_ub[i] - true_lb[i]))
        assert np.allclose(v.pdf(samples + true_ub[i]), 0)

    # Make sure a transformed Uniform variable works
    v = Variable(dist='Uniform(1e-8, 1e-2)', norm='log10')
    norm_bds = v.get_domain(norm=True)
    unnorm_bds = v.denormalize(norm_bds)
    samples_norm = v.sample(shape, norm=True)
    samples_unnorm = v.denormalize(samples_norm)
    assert np.all(np.logical_and(norm_bds[0] < samples_norm, samples_norm < norm_bds[1]))
    assert np.all(np.logical_and(unnorm_bds[0] < samples_unnorm, samples_unnorm < unnorm_bds[1]))


def test_normal_dist():
    shape = (10, 20, 50)
    true_means = [-5.2, 1, 2.7, 14.73, 100]
    true_stds = (np.random.rand(len(true_means)) * 0.1 + 0.05) * np.abs(true_means)
    variables = [Variable(dist=f'Normal({true_means[i]}, {true_stds[i]})') for i in range(len(true_means))]
    for i, v in enumerate(variables):
        # Plain normal distribution
        samples = v.sample(shape)
        x = np.linspace(*v.get_domain(), 1000)
        pdf = v.pdf(x)
        assert relative_error(np.mean(samples), true_means[i]) < 0.05
        assert relative_error(x[np.argmax(pdf)], true_means[i]) < 0.05

        # Normal distribution with transform
        v.update(norm='linear(2, 2)')
        samples = v.sample(shape, norm=True)
        x = v.denormalize(np.linspace(*v.get_domain(norm=True), 1000))
        pdf = v.pdf(x, norm=True)
        assert relative_error(np.mean(samples), v.normalize(true_means[i])) < 0.05
        assert relative_error(v.normalize(x[np.argmax(pdf)]), v.normalize(true_means[i])) < 0.05


def test_relative_and_tolerance_dist():
    shape = (10, 2)
    nominals = [1.1, -3, 1e6, 20.]
    dists = ['rel', 'tol']
    tols = [0.5, 1.1, 10]
    v = Variable()
    for nominal in nominals:
        for dist in dists:
            for tol in tols:
                v.update(nominal=nominal, dist=f'{dist}({tol})')
                samples = v.sample(shape)
                tol = tol if dist == 'tol' else (tol/100) * abs(nominal)
                bds = (nominal - tol, nominal + tol)
                assert np.all(np.logical_and(bds[0] < samples, samples < bds[1]))


def test_no_dist():
    shape = (10, 2)
    nominals = [-5.2, 1, 2.7, 14.73, 100]
    v = Variable()
    for i, nominal in enumerate(nominals):
        v.update(nominal=nominal)
        samples = v.sample(shape)
        x = np.linspace(*v.get_domain(), 100)
        pdf = v.pdf(x)
        assert np.allclose(samples, nominal)
        assert np.allclose(pdf, 1)
