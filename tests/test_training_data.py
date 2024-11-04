"""Test training data methods. Only `SparseGrid` methods are currently implemented."""
import itertools

import numpy as np

from amisc.compression import SVD
from amisc.training import SparseGrid
from amisc.utils import to_model_dataset
from amisc.variable import Variable, VariableList


def test_sparse_grid():
    """Test the `SparseGrid` data structure `get`, `set`, and `refine` methods."""
    def simple_model(inputs):
        y1 = inputs['x1']
        y2 = 10 ** (inputs['x2'])
        y3 = inputs['x3'] - 5
        return {'y1': y1, 'y2': y2, 'y3': y3}

    x_vars = VariableList([Variable('x1', distribution='U(0, 1)', norm='linear(2, 2)'),
                           Variable('x2', distribution='LU(1e-3, 1e-1)', norm='log10'),
                           Variable('x3', distribution='LN(0, 1)', norm='log10')])
    y_vars = VariableList(['y1', 'y2', 'y3'])
    domains = x_vars.get_domains(norm=True)
    weight_fcns = x_vars.get_pdfs(norm=True)
    grid = SparseGrid(collocation_rule='leja', knots_per_level=2)

    beta_list = list(itertools.product(*[range(3) for _ in range(len(x_vars))]))
    alpha_list = [()] * len(beta_list)
    design_list = []
    design_pts = []
    y_list = []

    # Refine the sparse grid
    for alpha, beta in zip(alpha_list, beta_list):
        new_idx, new_pts = grid.refine(alpha, beta, domains, weight_fcns)
        new_y = simple_model(new_pts)
        design_list.append(new_idx)
        design_pts.append(new_pts)
        y_list.append(new_y)
        grid.set(alpha, beta, new_idx, new_y)

    # Extract data from sparse grid and check values
    for i, (alpha, beta) in enumerate(zip(alpha_list, beta_list)):
        x_pts, y_pts = grid.get_by_coord(alpha, design_list[i])
        assert all([np.allclose(design_pts[i][var], x_pts[var]) for var in x_vars])
        assert all([np.allclose(y_list[i][var], y_pts[var]) for var in y_vars])


def test_imputer():
    """Test that imputation works for fixing missing data."""
    def simple_model(inputs, model_fidelity=(0,), frac1=0.03, frac2=0.03):
        err = 4 ** (-model_fidelity[0])
        y1 = inputs['x1'] * inputs['x3'] + err
        y2 = inputs['x2'] ** 2
        if np.random.rand() < frac1:
            y1[0] = np.nan
        if np.random.rand() < frac2:
            y2[0] = np.nan
        return {'y1': y1, 'y2': y2}

    x_vars = VariableList([Variable('x1', distribution='U(5, 10)', norm='minmax'),
                           Variable('x2', distribution='LU(1e-3, 1e-1)', norm='log10'),
                           Variable('x3', distribution='N(5, 1)', norm='zscore')])
    domains = x_vars.get_domains()
    weight_fcns = x_vars.get_pdfs()
    grid = SparseGrid()

    for idx in itertools.product(range(2), *[range(4) for _ in range(len(x_vars))]):
        alpha, beta = idx[:1], idx[1:]
        new_idx, new_pts = grid.refine(alpha, beta, domains, weight_fcns)
        new_y = simple_model(new_pts, alpha)
        grid.set(alpha, beta, new_idx, new_y)
        grid.impute_missing_data(alpha, beta)

    # Make sure reasonable values were imputed
    margin = 0.25
    yi_min = {'y1': -3, 'y2': 1}
    yi_max = {'y1': 3, 'y2': 9}
    for alpha in grid.yi_map:
        for coord, yi_dict in grid.yi_map[alpha].items():
            if any([np.any(np.isnan(yi)) for yi in yi_dict.values()]):
                try:
                    yi_dict_impute = grid.yi_nan_map[alpha][coord]
                    for var, yi in yi_dict_impute.items():
                        assert (1 - margin) * yi_min[var] < yi < (1 + margin) * yi_max[var]
                except KeyError as e:
                    raise ValueError(f'No imputed values found for alpha={alpha}, coord={coord}') from e

    for alpha in grid.yi_nan_map:
        for coord in grid.yi_nan_map[alpha]:
            assert any([np.any(np.isnan(yi)) for yi in grid.yi_map[alpha][coord].values()])


def test_field_quantity():
    """Test multidimensional inputs/outputs (and object arrays just for fun)"""
    rank = 4
    dof = 100
    svd = SVD(rank=rank, data_matrix=np.random.rand(dof, 50), coords=np.linspace(0, 1, dof))
    inputs = VariableList([Variable('x', distribution='U(0, 1)'), Variable('p', compression=svd, domain=(-20, 20)),
                           Variable('a', distribution='N(0, 1)')])
    weight_fcns = inputs.get_pdfs()
    domains = inputs.get_domains()
    def model(inputs, model_fidelity=(0,), frac=0.05):
        alpha = model_fidelity
        x = inputs['x']
        p = inputs['p']  # (N, dof)
        a = inputs['a']
        N, dof = p.shape
        y1 = np.sin(x) + np.sum(p ** 2, axis=-1) + a
        paths = np.random.choice([''.join(np.random.choice(list('abcdefg'), 5)) for _ in range(N * 2)], (N, 2))
        y2 = (p ** 2)[..., np.newaxis] * np.arange(alpha[0] + 1)  # (N, dof, alpha)
        if np.random.rand() < frac:
            y1[np.random.randint(N)] = np.nan
        return {'y1': y1, 'paths': paths, 'y2': y2}

    grid = SparseGrid(expand_latent_method='round-robin')
    for idx in itertools.product(range(2), *[range(3) for _ in range(len(inputs))]):
        alpha, beta = idx[:1], idx[1:]
        new_idx, new_pts = grid.refine(alpha, beta, domains, weight_fcns)
        new_y = model(to_model_dataset(new_pts, inputs)[0], model_fidelity=alpha)
        grid.set(alpha, beta, new_idx, new_y)
        grid.impute_missing_data(alpha, beta)

    for alpha in grid.yi_map:
        for coord, yi_dict in grid.yi_map[alpha].items():
            if np.any(np.isnan(yi_dict['y1'])):
                assert grid.yi_nan_map[alpha].get(coord) is not None
            assert np.atleast_1d(yi_dict['paths']).shape == (2, )
            assert np.atleast_1d(yi_dict['y2']).shape == (dof, alpha[0] + 1)
            assert len(coord[1]) == rank
