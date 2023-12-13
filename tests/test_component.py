import numpy as np
import matplotlib.pyplot as plt

from amisc.component import SparseGridSurrogate
from amisc.rv import UniformRV
from amisc.utils import ax_default


def test_sparse_grid(plots=False):
    """Simple cos test from Jakeman (2022)"""
    def model(x, alpha):
        alpha = np.atleast_1d(alpha)  # (1,)
        eps = (1/5) * 2.0**(-alpha[0])
        y = np.cos(np.pi/2 * (x + 4/5 + eps))
        return {'y': y}

    # Construct MISC surrogate from an index set
    Ik = [((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((2,), (0,)), ((1,), (1,)), ((0,), (2,)), ((1,), (2,)),
          ((2,), (1,)), ((2,), (2,))]
    x_vars = UniformRV(-1, 1)
    truth_alpha = (15,)
    comp = SparseGridSurrogate(x_vars, model, multi_index=Ik, truth_alpha=truth_alpha)
    N = 100
    xg = np.linspace(-1, 1, N).reshape((N, 1))
    yt = comp(xg, use_model=truth_alpha)
    y_surr = comp(xg)
    l2_error = np.sqrt(np.mean((y_surr - yt)**2)) / np.sqrt(np.mean(yt**2))
    assert l2_error < 0.1

    # Plot results for each fidelity of the MISC surrogate
    if plots:
        fig, axs = plt.subplots(3, 3, sharey='row', sharex='col')
        for alpha in range(3):
            for beta in range(3):
                ax = axs[2-alpha, beta]
                surr = comp.get_sub_surrogate((alpha,), (beta,), include_grid=True)
                s = f'$\hat{{f}}_{{{alpha}, {beta}}}$'
                ax.plot(xg, surr(xg), '--k', label=r'{}'.format(s), linewidth=1.5)
                s = f'$\hat{{f}}_{alpha}$'
                ax.plot(xg, model(xg, alpha)['y'], '--b', label=r'{}'.format(s), linewidth=2)
                ax.plot(xg, yt, '-r', label=r'$f$', linewidth=2)
                ax.plot(surr.xi, surr.yi, 'or')
                xlabel = r'$x$' if alpha == 0 else ''
                ylabel = r'$f(x)$' if beta == 0 else ''
                ax_default(ax, xlabel, ylabel, legend=True)

        fig.text(0.5, 0.02, r'Increasing surrogate fidelity ($\beta$) $\rightarrow$', ha='center', fontweight='bold')
        fig.text(0.02, 0.5, r'Increasing model fidelity ($\alpha$) $\rightarrow$', va='center', fontweight='bold', rotation='vertical')
        fig.set_size_inches(3 * 3, 3 * 3)
        fig.tight_layout(pad=3, w_pad=1, h_pad=1)
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(xg, yt, '-r', linewidth=2, label='Model')
        ax.plot(xg, y_surr, '--k', linewidth=1.5, label='MISC surrogate')
        ax_default(ax, r'$x$', r'$f(x)$', legend=True)
        plt.show()


# TODO: add tests for testing parallel execution, writing output files, and refining surrogate
