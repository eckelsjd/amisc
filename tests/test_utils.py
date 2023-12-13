import numpy as np
import matplotlib.pyplot as plt

from amisc.utils import approx_hess, batch_normal_sample, ax_default, get_logger, ndscatter


def test_hessian():
    """Test the Hessian of $f(x, y) = 2x^2 + 3xy^3$"""
    shape = (11, 12)
    dim = 2
    theta0 = np.random.rand(*shape, dim)
    fun = lambda theta: 2 * theta[..., 0:1]**2 + 3 * theta[..., 0:1] * theta[..., 1:2]**3

    H_tilde = approx_hess(fun, theta0)
    H_exact = np.empty(shape + (dim, dim))
    H_exact[..., 0, 0] = 4
    H_exact[..., 0, 1] = 9 * theta0[..., 1]**2
    H_exact[..., 1, 0] = 9 * theta0[..., 1]**2
    H_exact[..., 1, 1] = 18 * theta0[..., 0] * theta0[..., 1]

    assert np.allclose(H_tilde, H_exact, rtol=1e-3, atol=1e-3)


def test_sample():
    """Test 1d and 2d batch normal sampling"""
    dim = 2
    shape = (4, 5)
    N = 100000
    mean = np.random.rand(*shape, dim)
    cov = np.eye(dim) * 0.01
    samples = batch_normal_sample(mean, cov, N)
    assert samples.shape == (N, *shape, dim)
    assert np.allclose(np.mean(samples, axis=0), mean, rtol=1e-3, atol=1e-3)

    mean = np.random.rand()
    cov = 0.01
    samples = batch_normal_sample(mean, cov, N)
    assert np.isclose(mean, np.mean(samples, axis=0), rtol=1e-3, atol=1e-3)


def test_logging_and_plotting():
    """Test logging and plotting utils"""
    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    ax.plot(x, x, label='Hey there:)')
    ax_default(ax, 'X label here', 'Y label here', legend=True)
    logger = get_logger('tester', stdout=True)
    logger.info('Testing logger...')

    mean = np.array([5, -2])
    cov = np.array([[2, 0.4], [0.2, 1]])
    samples = np.random.multivariate_normal(mean, cov.T @ cov, size=100)
    yt = samples[:, 0] + samples[:, 1] ** 2
    ysurr = yt + np.random.randn(*yt.shape)
    err = np.abs(ysurr - yt) / np.abs(yt)
    ndscatter(samples, labels=[r'$\alpha$', r'$\beta$', r'$\gamma$'], plot='scatter', cmap='plasma',
              cb_norm='log', z=err)
