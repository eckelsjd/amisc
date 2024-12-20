"""Testing script for profiling the performance of `System.fit`."""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt

from amisc import Component, System, Variable


def io_bound_model(inputs, model_cost=1):
    time.sleep(model_cost)
    return {'y': -inputs['x1'] ** 3 + 2 * inputs['x3'] * np.cos(np.exp(inputs['x1']) * inputs['x2']) ** 2 +
                 np.sin(inputs['x3']) * inputs['x1'] * np.exp(inputs['x2'])}


def cpu_bound_model(inputs, model_cost=1):
    for _ in range(10**6 * model_cost):
        pass

    return {'y': -inputs['x1'] ** 3 + 2 * inputs['x3'] * np.cos(inputs['x2']) ** 2 +
                 np.sin(inputs['x3']) * inputs['x1']}


inputs = [Variable('x1', distribution='U(0, 1)'), Variable('x2', distribution='U(0, 1)'),
          Variable('x3', distribution='U(0, 1)')]
outputs = Variable('y')
max_tol = -np.inf


def profile_iteration_number():
    """Determine performance for various length iterations."""
    title_str = '------ Profiling iteration number ------'
    print(title_str)
    num_workers = 8
    model_cost = 0.5
    iterations = [5, 10, 20, 40]
    comp = Component(io_bound_model, inputs, outputs, data_fidelity=(3, 3, 3), model_cost=model_cost)
    surr = System(comp)

    t_serial = []
    t_vectorized = []
    t_parallel = []
    evals = []

    print(f'{"Iteration":>15s} {"Serial":>10s} {"Parallel":>10s} {"Vectorized":>10s} {"Model evals":>20s}')

    for iteration in iterations:
        model_evals = []

        # Serial
        surr.clear()
        surr['comp'].vectorized = False
        t1 = time.time()
        surr.fit(max_iter=iteration, max_tol=max_tol)
        t_serial.append(time.time() - t1)
        num_evals = np.cumsum(surr.get_allocation()[3])[-1]
        model_evals.append(num_evals)
        assert surr.refine_level == iteration

        # Parallel
        surr.clear()
        surr['comp'].vectorized = False
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            t1 = time.time()
            surr.fit(max_iter=iteration, max_tol=max_tol, executor=executor)
            t_parallel.append(time.time() - t1)
            num_evals = np.cumsum(surr.get_allocation()[3])[-1]
            model_evals.append(num_evals)
            assert surr.refine_level == iteration

        # Vectorized
        surr.clear()
        surr['comp'].vectorized = True
        t1 = time.time()
        surr.fit(max_iter=iteration, max_tol=max_tol)
        t_vectorized.append(time.time() - t1)
        num_evals = np.cumsum(surr.get_allocation()[3])[-1]
        model_evals.append(num_evals)
        assert surr.refine_level == iteration

        evals.append(int(model_evals[0]))

        print(f'{iteration:>15d} {t_serial[-1]:>10.2f} {t_parallel[-1]:>10.2f} {t_vectorized[-1]:>10.2f} '
              f'{str(tuple(map(int, model_evals))):>20}')

    print('-' * len(title_str) + '\n')

    fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
    iterations = np.atleast_1d(iterations)
    ax.plot(iterations, np.atleast_1d(evals) * model_cost, '-k', label='Model execution')
    ax.plot(iterations, np.atleast_1d(t_serial), '-r', label='Serial')
    ax.plot(iterations, np.atleast_1d(t_parallel), '-g', label='Parallel')
    ax.plot(iterations, np.atleast_1d(t_vectorized), '-b', label='Vectorized')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Duration of fit training (s)')
    ax.legend()
    ax.grid()
    plt.show()


def profile_model_cost():
    """Determine performance for various model costs."""
    title_str = '------ Profiling model cost ------'
    print(title_str)
    num_workers = 8
    costs = [0.1, 0.5, 1, 2, 4]  # in seconds
    iteration = 10
    comp = Component(io_bound_model, inputs, outputs, data_fidelity=(3, 3, 3))
    surr = System(comp)

    t_serial = []
    t_vectorized = []
    t_parallel = []
    evals = []
    print(f'{"Model cost (s)":>15s} {"Serial":>10s} {"Parallel":>10s} {"Vectorized":>10s} {"Model evals":>20s}')
    for cost in costs:
        model_evals = []
        surr['comp'].model_kwargs['model_cost'] = cost

        # Serial
        surr.clear()
        surr['comp'].vectorized = False
        t1 = time.time()
        surr.fit(max_iter=iteration, max_tol=max_tol)
        t_serial.append(time.time() - t1)
        num_evals = np.cumsum(surr.get_allocation()[3])[-1]
        model_evals.append(num_evals)
        assert surr.refine_level == iteration

        # Parallel
        surr.clear()
        surr['comp'].vectorized = False
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            t1 = time.time()
            surr.fit(max_iter=iteration, max_tol=max_tol, executor=executor)
            t_parallel.append(time.time() - t1)
            num_evals = np.cumsum(surr.get_allocation()[3])[-1]
            model_evals.append(num_evals)
            assert surr.refine_level == iteration

        # Vectorized
        surr.clear()
        surr['comp'].vectorized = True
        t1 = time.time()
        surr.fit(max_iter=iteration, max_tol=max_tol)
        t_vectorized.append(time.time() - t1)
        num_evals = np.cumsum(surr.get_allocation()[3])[-1]
        model_evals.append(num_evals)
        assert surr.refine_level == iteration

        evals.append(int(model_evals[0]))

        print(f'{cost:>15.1f} {t_serial[-1]:>10.2f} {t_parallel[-1]:>10.2f} {t_vectorized[-1]:>10.2f} '
              f'{str(tuple(map(int, model_evals))):>20}')

    print('-' * len(title_str) + '\n')

    costs = np.atleast_1d(costs)
    fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
    ax.plot(costs, np.atleast_1d(evals) * costs, '-k', label='Model execution')
    ax.plot(costs, np.atleast_1d(t_serial), '-r', label='Serial')
    ax.plot(costs, np.atleast_1d(t_parallel), '-g', label='Parallel')
    ax.plot(costs, np.atleast_1d(t_vectorized), '-b', label='Vectorized')
    ax.set_xlabel('Model cost (s)')
    ax.set_ylabel('Duration of fit training (s)')
    ax.legend()
    ax.grid()
    plt.show()


def profile_num_workers():
    """Determine performance for various numbers of workers."""
    title_str = '------ Profiling num workers ------'
    print(title_str)
    num_workers = [1, 2, 4, 8, 16]
    cost = 1
    iteration = 20
    comp = Component(io_bound_model, inputs, outputs, data_fidelity=(3, 3, 3), model_cost=cost)
    surr = System(comp)

    t_parallel = []
    evals = []

    # Serial
    print('Computing serial (for comparison)...')
    surr.clear()
    surr['comp'].vectorized = False
    t1 = time.time()
    surr.fit(max_iter=iteration, max_tol=max_tol)
    t_serial = np.atleast_1d([time.time() - t1] * len(num_workers))
    num_evals = np.cumsum(surr.get_allocation()[3])[-1]
    evals.append(num_evals)
    assert surr.refine_level == iteration

    # Vectorized
    print('Computing vectorized (for comparison)...')
    surr.clear()
    surr['comp'].vectorized = True
    t1 = time.time()
    surr.fit(max_iter=iteration, max_tol=max_tol)
    t_vectorized = np.atleast_1d([time.time() - t1] * len(num_workers))
    num_evals = np.cumsum(surr.get_allocation()[3])[-1]
    evals.append(num_evals)
    assert surr.refine_level == iteration


    print(f'{"Num workers":>15s} {"Serial":>10s} {"Parallel":>10s} {"Vectorized":>10s} {"Model evals":>20s}')
    for num_worker in num_workers:
        model_evals = []

        # Parallel
        surr.clear()
        surr['comp'].vectorized = False
        with ProcessPoolExecutor(max_workers=num_worker) as executor:
            t1 = time.time()
            surr.fit(max_iter=iteration, max_tol=max_tol, executor=executor)
            t_parallel.append(time.time() - t1)
            num_evals = np.cumsum(surr.get_allocation()[3])[-1]
            model_evals.append(num_evals)
            assert surr.refine_level == iteration

        print(f'{num_worker:>15d} {t_serial[-1]:>10.2f} {t_parallel[-1]:>10.2f} {t_vectorized[-1]:>10.2f} '
              f'{str(tuple(map(int, model_evals))):>20}')

    print('-' * len(title_str) + '\n')

    num_workers = np.atleast_1d(num_workers)
    fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
    ax.plot(num_workers, evals[0] * cost * np.ones(len(num_workers)), '-k', label='Model execution')
    ax.plot(num_workers, np.atleast_1d(t_serial), '-r', label='Serial')
    ax.plot(num_workers, np.atleast_1d(t_parallel), '-g', label='Parallel')
    ax.plot(num_workers, np.atleast_1d(t_vectorized), '-b', label='Vectorized')
    ax.set_xlabel('Number of workers')
    ax.set_ylabel('Duration of fit training (s)')
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == '__main__':
    profile_iteration_number()
    profile_model_cost()
    profile_num_workers()
