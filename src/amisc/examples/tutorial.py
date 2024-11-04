"""Examples to get started."""
# ruff: noqa: F841

def simple():
    # --8<-- [start:simple]
    import numpy as np

    from amisc import System

    def fun1(x):
        y = x * np.sin(np.pi * x)
        return y

    def fun2(y):
        z = 1 / (1 + 25 * y ** 2)
        return z

    system = System(fun1, fun2)
    system.inputs()['x'].update_domain((0, 1))
    system.outputs()['y'].update_domain((0, 1))

    system.fit()

    x_test = system.sample_inputs(10)
    y_test = system.predict(x_test)
    # --8<-- [end:simple]


def single_component():
    # --8<-- [start:single]
    from amisc import Component, System, Variable

    def fun(inputs):
        return {'y': inputs['x'] ** 2}

    x = Variable(distribution='U(-1, 1)')
    y = Variable()
    component = Component(fun, x, y, data_fidelity=(2,))
    system = System(component)

    system.fit()
    system.predict({'x': 0.5})  # {y: 0.25}
    # --8<-- [end:single]


def fire_sat():
    # --8<-- [start:fire_sat]
    from amisc.examples.models import fire_sat_system

    system = fire_sat_system()

    xtest = system.sample_inputs(100, use_pdf=True)     # --> (100, xdim)
    ytest = system.predict(xtest, use_model='best')     # --> (100, ydim)

    system.fit(test_set=(xtest, ytest), estimate_bounds=True)

    print(f'Inputs: {system.inputs()}')
    print(f'Outputs: {system.outputs()}')

    # Plots
    input_vars = ['H', 'Po']
    output_vars = ['Vsat', 'Asa']
    system.plot_allocation()
    system.plot_slice(input_vars, output_vars, show_model=['best', 'worst'], random_walk=True)
    # --8<-- [end:fire_sat]


def field_quantity():
    # --8<-- [start:field_qty]
    import numpy as np

    from amisc import Component, System, Variable, to_model_dataset
    from amisc.compression import SVD

    def my_model(inputs):
        """Compute a field quantity as a function of `x`."""
        field = np.sin(inputs['x'] * np.linspace(0, 1, 100))
        return {'f': field}

    dof = 100         # Number of degrees of freedom (i.e. size of the field qty)
    num_samples = 50  # Number of samples
    data_matrix = np.random.rand(dof, num_samples)
    field_coords = np.linspace(0, 1, dof)

    compression = SVD(rank=4, coords=field_coords, data_matrix=data_matrix)

    scalar = Variable('x', domain=(0, 1))
    field_qty = Variable('f', compression=compression)

    model = Component(my_model, inputs=scalar, outputs=field_qty, data_fidelity=(2,))
    system = System(model)

    system.fit(max_iter=10)

    xtest = system.sample_inputs(1000)
    ytest = system.predict(xtest)  # Will estimate the field qty in the latent/compressed space

    # Reconstruct the full field quantity
    ytest_reconstructed, _ = to_model_dataset(ytest, system.outputs())
    # --8<-- [end:field_qty]


if __name__ == '__main__':
    simple()
    single_component()
    fire_sat()
    field_quantity()
