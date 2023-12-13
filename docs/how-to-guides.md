## Specifying model inputs and outputs
Coming soon.

## Defining a component
Coming soon.

## Making a model wrapper function
The examples in the [tutorial](tutorials.md) use the simple function call signatures `ret = func(x)`, where `x` is an `np.ndarray` and 
`ret` is a dictionary with the required `y=output` key-value pair. If your model must be executed outside of Python
(such as in a separate `.exe` file), then you can write a Python wrapper function with the same call signature as above
and make any external calls you need inside the function (such as with `os.popen()`). You then pass the wrapper function
to `ComponentSpec` and `SystemSurrogate`.

!!! Note "Requirements for your wrapper function"
    - First argument `x` must be an `np.ndarray` of the model inputs whose **last** dimension is the number of inputs, i.e. `x.shape[-1] = x_dim`.
    - You can choose to handle as many other dimensions as you want, i.e. `x.shape[:-1]`. The surrogate will handle the same number 
      of dimensions you give to your wrapper function (so that `model(x)` and `surrogate(x)` are functionally equivalent). We recommend you handle at least 
      1 extra dimension, i.e. `x.shape = (N, x_dim)`. So your wrapper must handle `N` total sets of inputs at a time. The easiest way is to just 
      write a for loop over `N` and run your model for a single set of inputs at a time.
    - Your wrapper function must expect the `x_dim` inputs in a specific order according to how you defined your system. All
      system-level exogenous inputs (i.e. those in `system.exo_vars`) must be first and in the order you specified for
      `ComponentSpec(exo_in=[first, second, ...])`. All coupling inputs that come from the outputs of other models are next.
      Regardless of what order you chose in `ComponentSpec(coupling_in=[one, two, three,...]`, your wrapper **must** expect them
      in _sorted_ order according to `system.coupling_vars`. For example, if `system.coupling_vars = [a, b, c]` and 
      `comp = ComponentSpec(wrapper, coupling_in=[c, a], exo_in=[d, e], coupling_out=[f])`, then `x_dim = 4` and your `wrapper` function
      should expect the inputs in `x` to be ordered as `[d, e, a, c]`.
    - If you want to pass in model fidelity indices (see $\alpha$ in [theory](theory.md) for details), they must be in the form of a `tuple`,
      and your wrapper function should accept the `alpha=...` keyword argument. Specifying `alpha` allows managing a hierarchy of modeling fidelities, if applicable.
    - You can pass any number of additional positional arguments. Specify these with `ComponentSpec(model_args=...)`.
    - You can pass any number of keyword arguments. Specify these with `ComponentSpec(model_kwargs=...)`.
    - If you want to save and keep track of the full output of your model (i.e. if it writes result files to disk), then
      you can specify `ComponentSpec(save_output=True)`. When you do this, you must also specify `SystemSurrogate(..., save_dir='path/to/save/dir')`.
      You will then get a folder called `save_dir/amisc_timestamp/components/<your_model_name>`. This folder will be passed to your
      wrapper function as the keyword argument `output_dir=<your_model_dir>`. Make sure your `wrapper` accepts this keyword (no need to specify it in `ComponentSpec(model_kwargs=...)`; this is done automatically).
      You can then have your model write whatever it wants to this folder. You **must** then pass back the names of the files
      you created via `ret=dict(files=[your output files, ...])`. The filenames must be in a list and match the order in
      which the samples in `x` were executed by the model.
    - To assist the adaptive training procedure, you can also optionally have your model compute and return its computational cost via
      `ret=dict(cost=cpu_cost)`. The computational cost should be expressed in units of seconds of CPU time (not walltime!) for _one_ model evaluation.
      If your model makes use of `n` CPUs in parallel, then the total CPU time would be `n` times the wall clock time.
    - The return dictionary of your wrapper can include anything else you want outside of the three fields `(y, files, cost)` discussed here.
      Any extra return values will be ignored by the system.

!!! Example
    ```python
    def wrapper_func(x, *args, alpha=(0,), output_dir=None, **kwargs):
        print(x.shape)  # (..., x_dim)

        # Your code here, for example:
        output = x ** 2
        output_files = ['output_1.json', 'output2.json', ...]
        cpu_time = 42  # seconds for one model evaluation

        ret = dict(y=output, files=output_files, cost=cpu_time)

        return ret
    ```

!!! Warning
    Always specify the model at a _global_ scope, i.e. don't use `lambda` or nested functions. When saving to
    file, only a symbolic reference to the function signature will be saved, which must be globally defined
    when loading back from that save file.

## Putting it all together
Coming soon.