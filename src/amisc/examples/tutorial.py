"""Examples to get started."""


def single_component():
    # --8<-- [start:single]
    from amisc.system import SystemSurrogate, ComponentSpec
    from amisc.utils import UniformRV

    def fun(x):
        return dict(y=x ** 2)

    x = UniformRV(-1, 1)
    y = UniformRV(0, 1)
    component = ComponentSpec(fun)
    system = SystemSurrogate([component], x, y)

    system.fit()
    system.predict(0.5)  # 0.25
    # --8<-- [end:single]


def simple():
    # --8<-- [start:simple]
    import numpy as np

    from amisc.system import SystemSurrogate, ComponentSpec
    from amisc.utils import UniformRV

    def fun1(x):
        return dict(y=x * np.sin(np.pi * x))

    def fun2(x):
        return dict(y=1 / (1 + 25 * x ** 2))

    x = UniformRV(0, 1, 'x')
    y = UniformRV(0, 1, 'y')
    z = UniformRV(0, 1, 'z')
    model1 = ComponentSpec(fun1, exo_in=x, coupling_out=y)
    model2 = ComponentSpec(fun2, coupling_in=y, coupling_out=z)

    inputs = x
    outputs = [y, z]
    system = SystemSurrogate([model1, model2], inputs, outputs)
    system.fit()

    x_test = system.sample_inputs(10)
    y_test = system.predict(x_test)
    # --8<-- [end:simple]


def fire_sat():
    # --8<-- [start:fire_sat]
    import numpy as np
    import matplotlib.pyplot as plt

    from amisc.examples.models import fire_sat_system

    system = fire_sat_system()

    xtest = system.sample_inputs(100, use_pdf=True)     # --> (100, xdim)
    ytest = system(xtest, use_model='best')             # --> (100, ydim)
    use_idx = ~np.any(np.isnan(ytest), axis=-1)
    xtest = xtest[use_idx, :]
    ytest = ytest[use_idx, :]
    test_set = {'xt': xtest, 'yt': ytest}

    system.fit(max_iter=10, test_set=test_set, n_jobs=-1, num_refine=1000)

    print(f'Inputs: {system.exo_vars}')
    print(f'Outputs: {system.coupling_vars}')

    # Plots
    input_vars = ['H', 'Po']
    output_vars = ['Vsat', 'Asa']
    system.plot_allocation()
    system.plot_slice(input_vars, output_vars, show_model=['best', 'worst'], random_walk=True, N=10)
    plt.show()
    # --8<-- [end:fire_sat]


if __name__ == '__main__':
    single_component()
