"""Profile the prediction of a surrogate model with `py-spy`.

!!! Example
    First, install `py-spy` via `pip install py-spy`. Then, run the `generate_surrogate` function to
    generate a surrogate model. Finally, run the `predict_surrogate` function using `py-spy` externally via:

    ```shell
    py-spy record --rate 100 -o profile_predict.svg -- python profile_predict.py
    ```

    This will generate an interactive flame graph in `profile_predict.svg` to show where the most time is spent.

"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from amisc import Component, System, Variable


def compute_output(inputs):
    # Extract inputs from the dictionary
    x1 = inputs['x1']
    x2 = inputs['x2']
    x3 = inputs['x3']
    x4 = inputs['x4']
    x5 = inputs['x5']
    x6 = inputs['x6']
    x7 = inputs['x7']
    x8 = inputs['x8']
    x9 = inputs['x9']
    x10 = inputs['x10']

    # Compute the output array y with additional three- and four-way couplings
    y = (0.5 * x1 + 0.3 * x2 ** 2 - 0.1 * np.sin(x3)) + \
        (0.4 * x4 * np.exp(x5) - 0.2 * np.log(1 + x6)) + \
        (0.1 * x7 * x8 ** 2 - 0.3 * x9 * np.sqrt(x10 + 0.1)) + \
        (0.2 * np.cos(x1 * x5) * np.tan(x2 * x3)) + \
        (0.05 * x1 * x2 * x3) + \
        (0.1 * x4 * x5 * x6) + \
        (0.05 * x7 * x8 * x9 * x10) - \
        (0.05 * x1 * x5 * x9) + \
        (0.07 * np.sin(x2 * x4 * x6)) + \
        (0.08 * np.cos(x3 * x7 * x8 * x10))

    return {'y': y}


def generate_surrogate():
    inputs = [Variable(f'x{i+1}', distribution='U(0, 1)') for i in range(10)]
    outputs = [Variable('y')]
    comp = Component(compute_output, inputs, outputs, name='my_model', data_fidelity=(2,)*len(inputs),
                     vectorized=True)
    system = System(comp, root_dir='.', name='my_system')
    system.set_logger(stdout=True)

    with ProcessPoolExecutor() as executor:
        system.fit(max_iter=100, executor=executor, plot_interval=10, num_refine=500, max_tol=-np.inf)

    return system


def predict_surrogate(save_file):
    """Run several expensive surrogate predictions (and profile externally with py-spy)."""
    surr = System.load_from_file(save_file)
    sample_nums = [100, 500, 1000, 10000]

    for num_samples in sample_nums:
        inputs = surr.sample_inputs(num_samples)
        outputs = surr.predict(inputs)  # noqa: F841


if __name__ == '__main__':
    # Run this initially to generate the surrogate
    surr = generate_surrogate()
    iter_name = f'{surr.name}_iter{surr.refine_level}'
    save_file = surr.root_dir / 'surrogates' / iter_name / f"{iter_name}.yml"
    print(f'Surrogate save file at: {save_file}')

    # Run this with py-spy afterward using the generated surrogate file
    # save_file = Path('.') / 'amisc_2024-10-30T16.29.58/surrogates/my_system_iter100/my_system_iter100.yml'
    # predict_surrogate(save_file)
