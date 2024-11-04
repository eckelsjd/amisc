This guide will cover how to link models together and train a surrogate in `amisc`.

## Define a multidisciplinary system
The primary object for surrogate construction is the [`System`][amisc.system.System]. A `System` is constructed by passing all the component models:
```python
from amisc import System

def first_model(x1, x2):
    y1 = x1 * x2
    return y1

def second_model(y1, x3):
    y2 = y1 ** 2 + x3
    return y2

system = System(first_model, second_model)
```

More generally, you may pass the `Component` wrapper objects themselves with extra configurations to the system:
```python
from amisc import Component

system = System(Component(first_model, data_fidelity=(2, 2), ...),
                Component(second_model, data_fidelity=(3, 2), ...))
```

The `System` may also accept only a single component model as a limiting case. An MD system is compactly summarized by a directed graph data structure, with the nodes being the component models and the edges being the coupling variables passing between the components. You may view the system graph using `networkx` via:
```python
import networkx as nx

nx.draw(system.graph())
```

If you want to save a variety of surrogate build products and logs, set the `root_dir` attribute:
```python
system = System(..., root_dir='.')

system.root_dir = '/somewhere/else'  # alternatively
```

This will create a new `amisc_{timestamp}` save directory with the current timestamp under the specified directory, where all build products and save files will be written. The structure of the `amisc` root directory is summarized below:
```tree
amisc_2024-12-10T11.00.00/
    components/                     # folder for saving model outputs
        comp1/                      # outputs for 'comp1'
        comp2/                      # etc.
    surrogates/                     # surrogate save files
        system_iter0/
        system_iter1/
    amisc_2024-12-10T11.00.00.log   # log file
    plots.pdf                       # plots generated during training
```

!!! Note "Partial surrogates for an MD system"
    By default, the `System` will try to build a surrogate for _each_ component model in the system. If you don't want a surrogate to be built for a particular component model (e.g. if it's cheap to evaluate), then leave all `fidelity` options of the `Component` empty. This indicates that there is no way to "refine" your model, and so the `System` will skip building a surrogate for the component. You can check the `Component.has_surrogate` property to verify. During surrogate prediction, the underlying model function will be called instead for any components that do not have a surrogate.

## Train a surrogate
Surrogate training is handled by [`System.fit`][amisc.system.System.fit]. From a high level, surrogate training proceeds by taking a series of adaptive refinement steps until an end criterion is reached. There are three criteria for terminating the adaptive train loop:

- _Maximum iteration_ - train for a set number of iterations,
- _Runtime_ - train for at least a set length of time, then terminate at the end of the current iteration, or
- _Tolerance_ - when relative improvement between iterations is below a tolerance level.

For expensive models, it is _highly_ recommended to parallelize model evaluations by passing an instance of a [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor). At each sequential iteration, the parallel executor will manage evaluating the models on all new training data in a parallel loop.

It is also _highly_ recommended to pass an independent test set to evaluate the surrogate's generalization on unseen data. A test set is a tuple of two [Datasets][amisc.typing.Dataset]: one dataset for model inputs and one dataset for the corresponding model outputs. Test sets do not guide any aspect of the training -- they are just used as a metric for monitoring performance during training.

!!! Warning "Coupling variable bounds"
    Coupling variables are the inputs of any component model which are computed by and passed as outputs of an upstream component model. Since coupling variables are computed by a model, it may be difficult to know their domains _a priori_. When passing a test set to `System.fit()`, you may also set `estimate_bounds=True` to estimate all coupling variable bounds from the test set. Otherwise, you must manually set the coupling variable domains with a best guess for their expected ranges.

We leave the details of the adaptive refinement in the [AMISC journal paper](https://doi.org/10.1002/nme.6958), but you can view the logs and error plots during training to get an idea. Generally, at each iteration, the `System` loops over all candidate search directions for each component model, evaluates an "error indicator" metric that indicates potential improvement, and selects the most promising direction for more refinement. Once a direction is selected for refinement, new training points are sampled, the model is computed and stored, and the surrogate approximation is updated.

!!! Example
    ```python
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=8) as executor:
        system.fit(max_iter=100,            # max number of refinement steps
                   runtime_hr=3,            # run for at least 3 hrs and stop at end of current iteration
                   max_tol=1e-3,            # or terminate once relative improvement falls below this threshold
                   save_interval=5,         # save the surrogate to file every 5 iterations
                   plot_interval=1,         # plot error indicators every iteration
                   test_set=(xtest, ytest), # test set of unseen inputs/outputs
                   estimate_bounds=True,    # estimate coupling variable bounds from ytest
                   executor=executor)       # parallelize model evaluations
    ```

## Predict with a surrogate
Surrogate predictions are obtained using [`System.predict`][amisc.system.System.predict]. The surrogate expects to be called with a [Dataset][amisc.typing.Dataset] of model inputs, which is a dictionary with variable names as keys and corresponding numeric values. The values for each input may be arrays, for which the surrogate will be computed over all inputs in the array. You may use [`System.sample_inputs`][amisc.system.System.sample_inputs] to obtain a dataset of random input samples.

=== "Single input"
    ```python
    system = System(...)  # computes y1 = x1 * x2, for example
    
    inputs = { 'x1': 0.5, 'x2': 1.5 }
    output = system.predict(inputs)

    # will give { 'y1': 0.75 }
    ```

=== "Arrayed inputs"
    ```python
    system = System(...)  # computes y1 = x1 * x2, for example
    
    inputs = { 'x1': np.random.rand(100), 'x2': np.random.rand(100) }
    output = system.predict(inputs)

    # will give { 'y1': np.array(shape=100) }
    ```

!!! Note "Important"
    The input [Dataset][amisc.typing.Dataset] _must_ contain both normalized and compressed inputs (for field quantities) before passing to `System.predict`. This is because the surrogate was trained in the normalized space for `Variables` with the `norm` attribute, and in the latent space for field quantity `Variables` with the `compression` attribute. Likewise, return values from `predict` will be normalized and compressed outputs. See the [dataset conversion](#convert-datasets-for-model-or-surrogate-usage) section for more information.

You may also call `System.predict(..., use_model='best')` as an alias for calling the true component models instead of the surrogate (the inputs should still be normalized when passed in though -- they will get denormalized as needed under the hood).

Finally, there are two "modes" for evaluating the surrogate:

=== "Training mode"
    In training mode, only the active index sets are used in the MISC combination technique approximation (see [theory](../theory/overview.md) for details). Training mode uses only a subset of all available training data, and so its accuracy is generally worse than evaluation (or "testing") mode.
    ```python
    outputs = system.predict(inputs, index_set='train')
    ```

=== "Evaluation mode"
    In evaluation mode, all available training data is used, and so surrogate accuracy is generally higher than training mode. This is the default behavior of `predict`.
    ```python
    outputs = system.predict(inputs, index_set='test')  # default
    ```

## Evaluate surrogate performance
The `System` object provides three methods for evaluating the surrogate performance:

- [`test_set_performance`][amisc.system.System.test_set_performance] - computes the relative error between the surrogate and the true model on a test set,
- [`plot_slice`][amisc.system.System.plot_slice] - plots 1d slices of surrogate outputs over the inputs, and optionally compares to the true model,
- [`plot_allocation`][amisc.system.System.plot_allocation] - plots a bar chart that shows how computational resources were allocated during training.

!!! Example
    ```python
    system = System(...)  # define component models etc.
    system.fit()          # training

    # Evaluate surrogate performance
    rel_error = system.test_set_performance(xtest, ytest)
    system.plot_slice()
    system.plot_allocation()
    ```

## Saving to file
The `System` object provides two methods for saving and loading the surrogate from file:
```python
system.save_to_file('md_system.yml')
system = System.load_from_file('md_system.yml')
```

By default, these methods use the [YamlLoader][amisc.YamlLoader] class to read and write the `System` surrogate object from YAML files. If the `System.root_dir` property has been set, then save files will default to the `root_dir/surrogates` directory. If a save file is located within an `amisc_{timestamp}` directory, then the `root_dir` property will be set when loading from file.

!!! Note "YAML files"
    YAML files are a plain-text format, which allows easy inspection of the surrogate data saved in the file. You can also edit the surrogate properties directly in the file before loading back into memory. The save files closely mirror the format of [configuration files](config_file.md) and can be used as a template for future configurations.

## Convert datasets for model or surrogate usage
[Datasets][amisc.typing.Dataset] for passing input/output arrays have two formats:

=== "Model dataset"
    All values in the dataset are not normalized, and field quantities are in their full high-dimensional form (i.e. not compressed). This is how the model wrapper functions should expect their inputs to be formatted.
    ```python
    x = Variable(norm='log10', domain=(10, 100))     # a scalar
    f = Variable(compression=...)                    # a field quantity

    model_dataset = { 'x': 100, 'f': np.array([1, 2, 3, ...]) }
    system.predict(model_dataset, use_model='best', normalized_inputs=False)
    ```

=== "Surrogate dataset"
    All values in the dataset are normalized, and field quantities are split into `r` arrays with the special `LATENT` ID string, enumerated from `0` to `r-1`, where `r` is the rank of the compressed latent space for the field quantity.
    ```python
    x = Variable(norm='log10', domain=(10, 100))     # a scalar
    f = Variable(compression=...)                    # a field quantity

    surrogate_dataset = { 'x': np.log10(100), 'f_LATENT0': 1.5, 'f_LATENT1': 0.5, ..., 'f_LATENT{r}': 0.01 }
    system.predict(surrogate_dataset)
    ```

By default, [`System.predict`][amisc.system.System.predict] expects inputs in a normalized form for surrogate evaluation (but this may be toggled via the `normalized_inputs` flag). The [`System.sample_inputs`][amisc.system.System.sample_inputs] method will also return normalized/compressed inputs by default.

To convert between dataset formats (i.e. for comparing surrogate outputs to model outputs or vice versa), you may use the [`to_model_dataset`][amisc.utils.to_model_dataset] and [`to_surrogate_dataset`][amisc.utils.to_surrogate_dataset] utilities. These methods will use the `Variable` objects to perform the appropriate normalizations and compression/reconstruction during conversion.

!!! Example
    ```python
    from amisc import to_model_dataset, to_surrogate_dataset

    x = Variable(norm='log10', domain=(10, 100))     # a scalar
    f = Variable(compression=...)                    # a field quantity
    
    model_dataset = { 'x': 100, 'f': np.array([1, 2, 3, ...]) }

    surr_dataset, surr_vars = to_surrogate_dataset(model_dataset, [x, f])  # also returns the names of the latent variables
    model_dataset, coords = to_model_dataset(surr_dataset, [x, f])         # also returns grid coordinates for field quantities
    ```