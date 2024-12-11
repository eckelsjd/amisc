This guide will cover how to plug external models into the `amisc` framework using the `Component` class.

## The `Component` class
The `Component` class is a black-box wrapper around a single discipline model. For example, if you had a system composed of a compressor, combuster, and turbine for modeling a turbojet engine, then each of the three disciplines would be wrapped in their own `Component` object. Just like the underlying model, a `Component` needs a callable function, a set of input variables, and a set of output variables:
```python
from amisc import Component

def model(x):
    y = x ** 2
    return y

component = Component(model  = model,
                      inputs = 'x',
                      outputs= 'y')
```

For simple functions, the inputs and outputs can be inferred from the function signature:
```python
def my_model(x1, x2, x3):
    y1 = x1 + x2
    y2 = x2 + x3
    return y1, y2

component = Component(my_model)  # inputs and outputs are inferred

assert component.inputs == ['x1', 'x2', 'x3']
assert component.outputs == ['y1', 'y2']
```

More generally, a list of strings or `Variable` objects can be passed as `Component(my_model, inputs=[...], outputs=[...])`. If the model needs extra configurations or arguments, they may be passed as keyword arguments:

```python
def my_model(inputs, extra_config=None):
    # Etc.
    return outputs

component = Component(my_model, inputs, outputs, extra_config='My config value')
```

## Python wrapper functions
The `Component.model` attribute is where the user can link their external model into the `amisc` framework; we call these "Python wrapper functions". The model must be written as a simple callable function in Python; however, it may make any external calls it needs to run simulation software that may exist outside of Python, i.e. such as a binary executable on the local machine. Several examples are provided in the [tutorial](../tutorial.ipynb) and [elsewhere](../examples.md) in the documentation.

### Expected format
A wrapper function may be written in one of the two following formats:

=== "Packed inputs/outputs"
    The function accepts a single `dict` positional argument, from which all variables may be accessed by key. Similarly, the model returns a single `dict` from which all output variables may be accessed by key.
    ```python
    def my_model(input_dict):
        x1 = input_dict['x1']
        x2 = input_dict['x2']
        x3 = input_dict['x3']

        # Compute model

        output_dict = {}
        output_dict['y1'] = ...
        output_dict['y2'] = ...

        return output_dict
    ```

    This format is the default and should typically be preferred (since you don't have to worry about variable ordering).

=== "Unpacked inputs/outputs"
    The function accepts several positional arguments, one for each input variable. Similarly, the model returns all output variables in a tuple. This format may be preferred as it makes the call and return signatures more explicit; however, you must keep track of the ordering of arguments and return values.
    ```python
    def my_model(x1, x2, x3):
        # Compute model

        y1 = ...
        y2 = ...

        return y1, y2
    ```

    This format is used when the inputs/outputs are inferred from the function signature (i.e. not provided directly to `Component()`). You may manually set the `Component(call_unpacked=True, ret_unpacked=True)` attributes if you prefer this format. Note that it is then up to you to keep track of ordering if you later decide to change the component's inputs or outputs.

It is also possible to accept "packed" inputs but return "unpacked" outputs, or vice versa by specifying the `call_unpacked` and `ret_unpacked` attributes of `Component`.

!!! Note "Important note"
    We want to emphasize again that external models __do not__ have to be written in Python to be used with `amisc`. The user needs only a basic familiarity with Python to write a wrapper function in one of the above formats. There are many built-in and third-party libraries that enable Python to integrate well with many external ecosystems (e.g. `juliacall` for Julia programs, `matlab.engine` for MATLAB, `PyAnsys` for ANSYS, `requests` for web-based resources, and the very general built-in `subprocess.run` for executing arbitrary code on the local machine.) The availability of these resources makes Python the ideal "glue" language for integrating models from a variety of platforms.

!!! Warning "Using a global scope"
    Always specify the model at a _global_ scope, i.e. don't use `lambda` or nested functions. When saving to
    file, only a symbolic reference to the function signature will be saved, which must be globally defined (i.e. importable)
    when loading back from that save file.

### Calling the model
Regardless of how you prefer to format your model wrapper function, the `Component` class provides a consistent interface for calling the model function through the `Component.call_model()` function. As shown in the [surrogate guide](train_surrogate.md), the surrogate _always_ expects to be called in the "packed" format, i.e. by passing a `dict` whose keys are the input variables and whose values are the numeric values of those inputs (see the [Dataset][amisc.typing.Dataset] class). For example:
```python
from amisc import Component

comp = Component(...)

inputs = {'x1': 1.0, 'x2': 5.5}

outputs_surr  = comp.predict(inputs)                    # prediction with the surrogate
outputs_model = comp.call_model(inputs)                 # prediction with the underlying model

outputs_model = comp.predict(inputs, use_model='best')  # alias for call_model
```
As you can see, the `call_model` has the same signature as the `predict` function. We also provide the `use_model` alias within `predict` so that both surrogate and ground truth model predictions may be obtained using the exact same signature, regardless of how the wrapper function is written. The `call_model` interface also provides some convenience options for parallel and vectorized execution, which we cover later on [in this guide](#parallelize-the-model).

### Special model arguments
Apart from providing a consistent interface for calling the model, the `call_model` function will also inspect your wrapper function for special keywords and pass extra data to your function as requested. There are 5 special keyword arguments that your wrapper function may request:

- `model_fidelity` - will provide a set of fidelity indices to tune which physical fidelity of the model should be used (see the [model fidelity](#model-fidelities) section for details),
- `input_vars` - will provide the actual `Variable` objects from `Component.inputs` in case your model needs extra details about the inputs,
- `output_vars` - will similarly provide the `Variable` objects from `Component.outputs`,
- `output_path` - will give your function a `Path` to a directory where it can write model outputs to file (e.g. if you have lots of extra save data you want to keep). This would be useful if you want to keep the context of each model evaluation with respect to the `amisc` surrogate, so that you might later construct a new surrogate for outputs you saved to file. This is similar to the `tmp_path` within the `pytest` framework.
- `{{ var }}_coords` - for a given "var" name, this will pass the compression grid coordinates for a given field quantity input variable.

You may additionally pass any extra keyword arguments on to the model directly through `call_model(inputs, **kwargs)`.

!!! Example
    ```python
    def my_model(inputs, model_fidelity=(0,), output_path=None, input_vars=None, x_coords=None, config=None):
        x = inputs['x']                           # get the input values for 'x'

        if x_coords is None:                      # read compression grid coordinates for the 'x' field quantity
            x_coords = np.linspace(0, 1, 100)     # default

        delta_t = 1e-8 / (model_fidelity[0] + 1)  # set the time step based on model fidelity level

        if x < 0:
            x = input_vars['x'].get_nominal()     # set a threshold for x using info from the `Variable` objects

        # Compute model
        y = ...

        if output_path is not None:               # write outputs to file
            with open(output_path / 'my_outputs.txt', 'w') as fd:
                fd.writelines(f'y = {y}')

        return {'y': y, 'output_path': 'my_outputs.txt'}

    comp = Component(my_model, inputs=Variable('x', nominal=1), outputs=Variable('y'))
    pred = comp.call_model({'x': 0.5}, model_fidelity=(1,), config={'extra': 'configs'})
    ```

### Special model return values
In addition to special keyword arguments, your wrapper function may also return special values. There are 4 values that may be returned by the wrapper function:

- `{{ var }}` - for a given output "var" name -- these are just the usual numeric return values that match the variables in `Component.outputs`,
- `{{ var_coords }}` - for a given field quantity output "var" name -- these are the grid coordinates corresponding to the locations of the "var" output field quantity (returned shape is $(N, D)$ where $N$ is the number of points and $D$ is the number of dimensions),
- `model_cost` - a best estimate for total model evaluation time (in seconds of CPU time). For example, if your model makes continuous use of 16 CPUs and takes 1 hour to evaluate one set of inputs, then you would return `16 * 3600` as the model cost. This metric is used by `amisc` to help evaluate which are the most "effective" model evaluations, i.e. more expensive models are penalized greater than cheaper models.
- `output_path` - if your function _requested_ `output_path` and subsequently wrote some data to file, then you can _return_ the new file name as `output_path` to keep track of the file's location within the `Component's` internal data storage.

Your wrapper function may also return any extra data that you wish to store within the `Component` -- `amisc` will keep track of this data but will not use it for any purpose. You might do this if you wish to return to old model evaluations later on and view data even if you did not explicitly build a surrogate for the data. This extra data will be stored in `numpy` object arrays, so the data does not have to be strictly numeric type -- it could even be custom object types of your own. Field quantities will also be stored in `numpy` object arrays.

!!! Example
    ```python
    def my_model(inputs, output_path=None):
        t1 = time.time()    # time the model evaluation
        # Compute the model
        y1 = ...
        y2 = ...

        t2 = time.time()

        if output_path is not None:
            with open(output_path / 'my_output.txt', 'w') as fd:
                fd.writelines(f'I want to store some output data via file here')

        extra_data = {'store': 'extra', 'data': 'here'}

        return {'y1': y1, 'y2': y2, 'model_cost': t2 - t1, 'output_path': 'my_output.txt', **extra_data}
    ```

## Model fidelities
As we saw in the [special arguments](#special-model-arguments) section, your wrapper function may request the `model_fidelity` keyword argument. This will pass a tuple of integers (what we call a _multi-index_) to your function, for which you can use to "tune" the fidelity level of your model. For example, if you build a `Component` and specify a tuple of two integers: `Component(..., model_fidelity=(3, 2))`, then the "maximum" fidelity of your model is fully specified by the numbers $(3, 2)$. You can use these integers however you want to adjust model fidelity, e.g. `num_grid_pts = 100 * (model_fidelity[0] + 1)` would use the first integer to tune how many grid points are used in a simulation, with higher integers corresponding to more grid points, and therefore higher numerical accuracy. Your wrapper function should then request `model_fidelity` and use the multi-index as desired:
```python
def my_model(inputs, model_fidelity=(0, 0)):
    num_grid_pts = 100 * (model_fidelity[0] + 1)      # controls number of grid points
    time_step = 1e-8 * (1 / (model_fidelity[1] + 1))  # controls time step size

    # etc.

    return outputs
```
During surrogate training, your wrapper function will then be passed tuples of varying model fidelity (from $(0, 0)$ up to $(3, 2)$ in the example). If you do not request `model_fidelity`, then your wrapper function will not be passed any fidelity indices, and will instead be treated as a single fidelity model.

Apart from `model_fidelity`, you may also specify similar tuples of integers for `data_fidelity` and `surrogate_fidelity`, which correspond to fidelity levels for training data and the surrogate method, respectively. These indices are used internally by `amisc` during training and are not passed to your wrapper function like `model_fidelity`. Instead, together they tune the amount of training data collected and the refinement level of the surrogate method (e.g. number of hidden layers, weights, etc.) As such, their usage is dependent on the specific underlying methods for training data collection and surrogate evaluation.

The [tutorial](../tutorial.ipynb) provides more detail on how to interpret and use model fidelities. For the purposes of this guide, we summarize the fidelity options for the `Component` class in the example below:
```python
comp = Component(..., model_fidelity=(3, 2),       # the maximum model fidelity indices
                      data_fidelity=(2, 3, 1),     # the maximum training data fidelity indices
                      surrogate_fidelity=())       # the maximum surrogate fidelity indices
```

By default, the only training data method available is the `SparseGrid` -- the sparse grid method uses one index in `data_fidelity` per input, so that `len(data_fidelity) == len(inputs)`. Each index determines the maximum training data allowed along each input dimension. For a `data_fidelity` of $\beta$ for example, each input $x_i$ for $i=0\dots N$ has a maximum number of training points of $2\beta_i + 1$, which for $\beta = (2, 3, 1)$ means a max grid size of $(4, 7, 3)$. These options are configurable, and more information can be found at [SparseGrid][amisc.training.SparseGrid].

By default, the only surrogate method available is the `Lagrange` -- Lagrange polynomial interpolation does not use any method for tuning its own fidelity, so `surrogate_fidelity` may be left empty. See more details at [Lagrange][amisc.interpolator.Lagrange].

We frequently refer to `model_fidelity` by the symbol "$\alpha$", and to the combination of `data_fidelity` and `surrogate_fidelity` as "$\beta$". For a given pair of $(\alpha, \beta)$, the `Component` surrogate can be trained by "activating" the index pair via:
```python
comp = Component(my_model, inputs, outputs, model_fidelity=..., data_fidelity=..., surrogate_fidelity=...)

alpha, beta = (2, 1), (3, 2)      # for example
comp.activate_index(alpha, beta)  # 'train' the surrogate for this index pair
```

By calling `activate_index`, we will move $(\alpha, \beta)$ into the component's "active" index set, and then search for and compute model evaluations for all the "neighboring" index pairs. For most users, this will be done automatically on a call to `System.fit()`, but we provide this description here for completeness. To get a better understanding of "active" and "neighboring" index sets, please see the [theory](../theory/overview.md) section.

## Storing and retrieving training data
The `Component.training_data` attribute provides the method by which training data is sampled (i.e. experimental design), stored, and retrieved. Currently, the only available method is the [SparseGrid][amisc.training.SparseGrid], which is set by default. For the most part, all aspects of training data manipulation is handled behind the scenes by the `Component` class. You may configure the training data method via:
```python
from amisc.training import SparseGrid

comp = Component(..., training_data=SparseGrid(collocation_rule='leja'))
```

New training data methods may be created by implementing the `amisc.training.TrainingData` interface. Data storage and retrieval is generally guided by the $(\alpha, \beta)$ fidelity indices described in the [fidelity](#model-fidelities) section. The `Component` class provides the `get_training_data` method to extract all input/output data associated with a given $(\alpha, \beta)$ pair:
```python
alpha, beta = (2, 1), (3, 2)      # for example
xtrain, ytrain = component.get_training_data(alpha, beta)
```

By default `get_training_data` will return the training data associated with the highest-fidelity pair of $(\alpha, \beta)$. After fitting a surrogate, the user may wish to extract the training data for each fidelity level and attempt construction of other surrogates to compare performance; this is made possible by the `get_training_data` interface.

As one last note, if your model returns field quantity data, this will be stored along with its compressed latent space representation. By default, both representations of the field quantity are returned by `get_training_data`.

## Specifying a surrogate method
The `Component.interpolator` attribute provides the underlying surrogate "interpolation" method, i.e. the specific mathematical relationship that approximates the model's outputs as a function of its inputs. In this sense, we use the terms "interpolator" and "surrogate" interchangeably to mean the underlying approximation method -- the `Component.interpolator` does not necessarily have to "interpolate" the output by passing through all the training data directly. The naming convention mostly arises from the usage of polynomial interpolation in sparse grids.

Currently, the only available interpolation method is the [Lagrange][amisc.interpolator.Lagrange] polynomial interpolation, which is set by default. Multivariate Lagrange polynomials are formed by a tensor-product of univariate Lagrange polynomials in each input dimension, and integrate well with the `SparseGrid` data structure. Lagrange polynomials work well up to an input dimension of around 12-15 for sufficiently smooth functions. More details on how they work can be found in the [theory](../theory/polynomials.md) section.

You may configure the interpolation method via:
```python
from amisc.interpolator import Lagrange

comp = Component(..., interpolator=Lagrange())
```

New interpolation/surrogate methods may be created by implementing the `amisc.interpolator.Interpolator` interface.

## Parallelize the model
A large advantage of the `Component.call_model()` interface is the ability to evaluate multiple input samples with the model in parallel. If your model wrapper function only processes one set of inputs at a time (i.e. only single values in the `inputs` dictionary), then calling the model through `call_model` will automatically allow handling arrays of inputs. For example:

=== "Single input"
    ```python
    def my_model(inputs):
        x1 = inputs['x1']       # single float
        x2 = inputs['x2']       # single float

        return {'y1': x1 * x2}  # single float

    output = my_model({ 'x1': 0.5, 'x2': 1.5 })

    # will give { 'y1': 0.75 }
    ```

=== "Arrayed inputs"
    ```python
    def my_model(inputs):
        x1 = inputs['x1']       # single float
        x2 = inputs['x2']       # single float

        return {'y1': x1 * x2}  # single float

    comp = Component(my_model, ['x1', 'x2'], 'y1')
    output = comp.call_model({ 'x1': np.random.rand(100), 'x2': np.random.rand(100) })

    # will give { 'y1': np.array(shape=100) }
    ```

By default, this will run the input samples in a serial loop, which will not provide any speedup, but does allow your single-input wrapper function to be used with arrays of inputs.

To parallelize the model over the input arrays, you may pass an instance of an `Executor` to `call_model` (i.e. something that implements the built-in Python `concurrent.futures.Executor` interface -- Python itself provides the `ProcessPoolExecutor` and `ThreadPoolExecutor` which use parallel process- or thread-based workers). The popular `mpi4py` library also provides an `MPIExecutor` to distribute parallel tasks over MPI-enabled workers. You may similarly pass an instance of `Executor` to any functions where you wish to parallelize model evaluations (such as `System.fit()` for training the surrogate).

Finally, if your wrapper function can handle arrayed inputs on its own, then you may set `Component.vectorized=True`. Input dictionaries will then be passed to your wrapper function with `np.ndarrays` as values for each input variable rather than scalar `floats`. For example, you can take advantage of `numpy` vectorization to directly manipulate two arrays of inputs rather than looping over each element (e.g. `y = x1 * x2` rather than `y = [x1[i] * x2[i] for i in range(N)]`).

If you do use `vectorized=True` and you're also using `model_fidelity`, you should expect `model_fidelity` as an `np.ndarray` of shape `(N, R)`, where `N` is the loop shape of the inputs and `R` is the number of fidelity indices; usually you would just expect a tuple of length `R` when `vectorized=False`.

=== "Serial"
    ```python
    def my_model(inputs):
        x1 = inputs['x1']   # float
        x2 = inputs['x2']   # float
        return { 'y1': x1 * x2 }

    comp = Component(my_model, ['x1', 'x2'], 'y1')

    in  = { 'x1': np.random.rand(100), 'x2': np.random.rand(100) }
    out = comp.call_model(in)
    ```

=== "Parallel"
    ```python
    def my_model(inputs):
        x1 = inputs['x1']   # float
        x2 = inputs['x2']   # float
        return { 'y1': x1 * x2 }

    comp = Component(my_model, ['x1', 'x2'], 'y1')

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        in  = { 'x1': np.random.rand(100), 'x2': np.random.rand(100) }
        out = comp.call_model(in, executor=executor)
    ```

=== "Vectorized"
    ```python
    def my_model(inputs):
        x1 = inputs['x1']   # np.ndarray (100,)
        x2 = inputs['x2']   # np.ndarray (100,)
        return { 'y1': x1 * x2 }

    comp = Component(my_model, ['x1', 'x2'], 'y1', vectorized=True)

    in  = { 'x1': np.random.rand(100), 'x2': np.random.rand(100) }
    out = comp.call_model(in)
    ```

Generally, the surrogate predictions (e.g. `Component.predict`) make use of `numpy` vectorization to handle arrayed inputs, so `call_model` again acts as a consistent interface for calling the underlying model with the same signature as the surrogate.

!!! Note "Loop dimensions"
    If you pass a scalar input array of arbitrary shape (say $(100, 2, 5)$ for example), `Component.call_model()` treats all axes in the array as "loop dimensions". So effectively, all the axes will be flattened to a single 1d array of length `N = np.prod(input.shape)`, which would be $N=1000$ for an input shape of $(100, 2, 5)$. Then, `call_model` will loop over $N$, passing each input sample to the underlying model (unless `vectorized=True`). The outputs will be reshaped back to the original input "loop" shape before returning. If you do pass an input array for one variable, you should pass the same shape array for all other inputs, or at least an array that is broadcastable to the full loop shape. See [`numpy` broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) rules.

    The situation is a bit more complicated for non-scalar inputs (i.e. field quantities). You may still pass arrays of field quantities as inputs, so that they are looped over in the same fashion as scalar inputs. However, which axes are a "loop shape" vs. what axes are part of the field quantity's shape itself must be inferred or indicated. Generally, as long as you provide at least one scalar input, then the loop shape is known and the first leading axes of the field quantity that match the loop shape will be used as such, while the rest are assumed to be the field quantity's own shape. For example:
    ```python
    x = np.random.rand(100, 2)      # scalar
    f = np.random.rand(100, 2, 20)  # field quantity with shape (20,)
    ```

    Clearly, the loop shape is (100, 2) and we can infer the field quantity shape of (20,). If the field quantity were provided by itself, then you should also provide `f_coords` to `call_model()` so that the field quantity shape for the `f` variable can be obtained from the shape of its grid coordinates. See the [field quantity](variables.md) documentation for more details.

## `Component` configuration summary
Here is a summary of all configurations that may be passed to `Component` by example:
```python
def my_model(inputs, model_fidelity=(0, 0), **extra_kwargs):
    x1 = inputs['x1']
    x2 = inputs['x2']
    x3 = inputs['x3']

    return {'y1': ..., 'y2': ...}

comp = Component(model=my_model,                # The Python wrapper function of the component model
                 inputs=['x1', 'x2', 'x3'],     # List of input variables (str or Variable objects)
                 outputs=['y1', 'y2'],          # List of output variables (str or Variable objects)
                 model_fidelity=(2, 3),         # Physical model fidelity multi-index
                 data_fidelity=(2, 2, 1),       # Data fidelity multi-index (length matches len(inputs) for SparseGrid)
                 surrogate_fidelity=(),         # Surrogate fidelity multi-index (empty for Lagrange)
                 training_data=SparseGrid(),    # Collocation/design and storage method for training data
                 interpolator=Lagrange(),       # Underlying surrogate/interpolation method
                 vectorized=True,               # If wrapper function handles arrayed inputs
                 call_unpacked=False,           # Call signature of   `func(x1, x2, ...)`  if True
                 ret_unpacked=False,            # Return signature of `return y1, y2, ...` if True
                 name='My component',           # ID for log messages and referencing
                 **extra_kwargs)                # Extra keyword arguments to pass to wrapper function
```

A few important things to keep in mind:

- The first three arguments (`model`, `inputs`, `outputs`) must always be specified (or at least inferred for inputs/outputs) and can be passed as positional arguments.
- If all of the `fidelity` options are left empty, then no surrogate will be built.
- The wrapper function must request `model_fidelity` as a keyword argument if you specify it in the `Component`.
- Data and surrogate fidelity usage are dependent on `training_data` and `interpolator`, respectively. The length of `data_fidelity` should match the number of inputs for `SparseGrid` training data.
- Setting `vectorized=True` means the wrapper function should handle arrayed inputs (as well as arrayed `model_fidelity` if applicable).
- The "packed" call and return signatures are preferred (i.e. the default is `call_unpacked=False` and `ret_unpacked=False`) -- the wrapper function should receive and return dictionaries, where the keys are the variables with their corresponding values.
- The wrapper function should additionally return the special variables `{{ var }}_coords` to provide the grid coordinates for field quantity outputs.


## Viewing internal data
There are a few use cases for wanting to manipulate the internal data structures of a `Component`, so for completeness, we summarize the data structures here. Components store data with respect to the $(\alpha, \beta)$ fidelity indices; see the [model fidelity](#model-fidelities) section for more details.

### Index sets
An index set is a set of concatenated multi-indices $(\alpha, \beta)$. For example, $\mathcal{I}=\{ ((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((1,), (1,)) \}$ is the set of all combinations of $\alpha$ and $\beta$ ranging from 0 to 1.

A `Component` has two index sets: an `active_set` and a `candidate_set`. The active set is the set of fidelity levels that are currently being used in the surrogate approximation. The candidate set is all fidelity levels that are forward neighbors of the active set, e.g. $(1, 0)$ and $(0, 1)$ are forward neighbors of $(0, 0)$. While training the surrogate, only the active set is used for prediction. After training, both active and candidate sets are used together for prediction.

### MISC trees
For every $(\alpha, \beta)$ multi-index pair, the component stores four items in a tree-like structure:

- _interpolator state_ - the state of the underlying interpolator (e.g. weights, biases, coefficients, etc.). Since a single $(\alpha, \beta)$ pair corresponds to a single set of training data, the interpolator state is the corresponding surrogate trained on this data.
- _cost_ - the computational expense incurred by adding $(\alpha, \beta)$ to the surrogate approximation, i.e. the cost of evaluating the model on the new training data.
- _train coefficients_ - the combination technique coefficients for the $(\alpha, \beta)$ surrogate under the active index set (see [theory](../theory/overview.md) for more details).
- _test coefficients_ - the combination technique coefficients for the $(\alpha, \beta)$ surrogate under both the active and candidate index sets (see [theory](../theory/overview.md) for more details).

Additionally, the computational expense (seconds of CPU time for one evaluation) of each $\alpha$ fidelity level is stored. The internal data structures of the `Component` are summarized below:
```python
comp = Component(...,
                 active_set=IndexSet(...),          # "active" index set during training
                 candidate_set=IndexSet(...),       # "candidate" index set (forward neighbors of the active set)
                 misc_states=MiscTree(...),         # interpolator states (weights, biases, etc.)
                 misc_costs=MiscTree(...),          # cost incurred to add each multi-index pair
                 misc_coeff_train=MiscTree(...),    # combination technique coeffs (active)
                 misc_coeff_test=MiscTree(...),     # combination technique coeffs (active+candidate)
                 model_costs=dict(...))             # computational expense of each model fidelity level
```

### Overriding the data structures
All of the `Component` data structures are managed internally and do not need to be set by the user. However, the `Component.predict()` function accepts an `index_set` and a `misc_coeff` argument so that the user may predict with the surrogate using an alternate set of training data. In this case, the user would maintain their own `IndexSet` and `MiscTree` structures outside the `Component` and pass them as `Component.predict(..., index_set=index_set, misc_coeff=misc_coeff)`.

For example, during surrogate training via `System.fit()`, the component surrogate is called with a modified index set while adaptively searching for the next best $(\alpha, \beta)$ pair to add to the surrogate approximation. See the [System.refine][amisc.system.System.refine] documentation.
