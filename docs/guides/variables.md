Variables are the basic objects used as inputs or outputs in a model. In this guide, we will learn how to construct and use variables within `amisc`.

## Construct a variable
In its most basic form, a variable is just a placeholder with a name, just like $x$ in the equation $y=x^2$.

```python
from amisc import Variable

x = Variable()     # implicitly named 'x'
y = Variable('y')  # explicitly named 'y'
```

Variables can also have several descriptive attributes assigned to them.
```python
x = Variable(name='x1',                         # the main identification string of the variable
             nominal=1,                         # a nominal value
             description='My first variable',   # a lengthier description
             units='rad/s',                     # units
             tex='$x_1$',                       # a latex representation (for plotting/displaying)
             category='calibration')            # for further classification (can be anything)
```

The variable's `name` is the key identifier of the variable, and allows the variable to be treated symbolically as a string. For example:
```python
x = Variable('x')
assert x == 'x'

d = {x: 'You can use the variable as a key in hash structures'}
assert d[x] == d['x']
```

In addition, a useful data structure for lists of `Variables` is the `VariableList`:
```python
from amisc import VariableList, Variable

var_list = VariableList(['a', 'b', 'c'])

assert var_list['a'] == 'a'                   # can use 'dict'-like access of variables
assert isinstance(var_list['a'], Variable)    # stores the actual Variable objects
assert var_list[2] == var_list['c']           # can also use normal 'list' indexing
```

An important attribute of `Variables` in the context of `amisc` is their domain, which must be defined when building surrogates:
```python
x = Variable('x', domain=(0, 1))  # domain over which surrogates will be built
```

There are three more attributes of variables that we will cover in the next sections: normalization, distributions (for random variables), and compression (for field quantities).

## Normalization
In the context of surrogates, it is sometimes advantageous to approximate over a transformed, or normalized input space. For example, a variable defined over the domain $x\in (0.001, 100)$ covers many orders of magnitude, which may be difficult to directly approximate using a polynomial surrogate. There are four basic normalizations provided by `amisc`:
```python
from amisc.transform import Log, Linear, Minmax, Zscore

log = Log((10, 0))                  # base 10 log with 0 offset
linear = Linear((0.5, 1))           # slope of 0.5 and offset of 1
minmax = Minmax((-20, 20, 0, 1))    # scale from (-20, 20) -> (0, 1)
zscore = Zscore((5, 2))             # (x - mu) / sigma
```

These may also be specified as an equivalent string representation. The transform method should be passed as the `norm` attribute of the variable:
```python
for norm in ['log10', 'linear(0.5, 1)', 'minmax', 'zscore']:
    x = Variable(norm=norm)
```

Values can then be normalized or denormalized directly by the variable:
```python
import numpy as np

x = Variable(norm='log10')
values = 10 ** (np.random.rand(20))

assert np.allclose(x.denormalize(x.normalize(values)), values)
```

When a variable has a `norm`, the surrogate will select new training points in the transformed space and also compute the approximation on normalized inputs. If a variable is an output and has a `norm`, then the surrogate will fit the approximation to the normalized output.

!!! Example "Building a surrogate in normalized space"
    Consider the variable defined as:
    ```python
    x = Variable(norm='log10', domain=(1e-3, 1e2))
    ```
    The surrogate will construct an approximation over the transformed domain $(-3, 2)$. When predicting with the surrogate, inputs will automatically have the same transform applied $\tilde{x} = \log_{10}(x)\in(-3, 2)$ before computing the surrogate.

New transforms can be created by extending the `amisc.transform.Transform` base class. In addition, multiple transforms can be applied in series by passing a list of transforms to the `norm` attribute. For example, `x = Variable(norm=['log10', 'minmax'])` will apply a `minmax` transform over the `log10` space of `x`.

## Random variables
A common use of surrogates is to permit propagating uncertain random variable inputs through a complicated simulation to quantify output uncertainty or to calibrate the model parameters. To this end, a `Variable` can be given a PDF through the `distribution` attribute. Several common PDFs are provided in `amisc.distribution`.
```python
uniform     = Variable(distribution='U(0, 1)')
normal      = Variable(distribution='N(0, 1)')
log_uniform = Variable(distribution='LU(1e-3, 1e2)')
log_normal  = Variable(distribution='LN(-2, 1)')
```

With a distribution, variable's can sample from the PDF or evaluate the PDF of values under the distribution:
```python
x = Variable(distribution='N(0, 1)')

samples = x.sample(1000)
pdfs    = x.pdf(samples)
```

When a variable has a distribution, the surrogate will select new training points during `fit()` that are clustered closer to areas of greater weight. New distributions can be created by extending the `amisc.distribution.Distribution` base class.

## Field quantities
By default, all variables are treated as _scalar_ quantities. However, it is sometimes possible to have high-dimensional variables, such as the solution of a simulation on a PDE mesh -- we refer to these variables as "field quantities". For field quantities to be useful in the context of `amisc` surrogates, we must be able to "compress" them to lower dimension such that we can effectively build surrogate approximations in an appropriate low-dimensional "latent" space.

To this end, a field quantity is defined by giving a `compression` attribute to a variable. A compression method must:

- define a set of coordinates on which the field quantity exists (i.e. the Cartesian points from a simulation mesh grid),
- define a "map" that both _compresses_ field quantity data into the latent space and _reconstructs_ the full field quantity back from the latent space, and
- have a predetermined size (or "rank") of the latent space.

Compression coordinates should be an array of shape $(N, D)$, where $N$ is the total number of grid points and $D$ is the Cartesian dimension (i.e. 1d, 2d, etc.). A single field quantity `Variable` may contain several QoIs on the same grid coordinates, so that the total number of "degrees of freedom" (DoF) of the variable is equal to $N\times Q$, where $Q$ is the number of QoIs.

For example, say a simulation outputs the $x, y, z$ components of velocity on an unstructured mesh of 1000 nodes. We might define a velocity field quantity as:
```python
vel = Variable('velocity', compression=dict(coords=sim_coords,
                                            fields=['ux', 'uy', 'uz'],
                                            method='svd'))

print(sim_coords.shape)                 # (num_pts, dim)
assert vel.compression.dof == 1000 * 3  # (num_pts * num_qoi)
```
for some predefined set of Cartesian `sim_coords` that we extracted from our simulation. Currently, SVD is the only available compression method, but other methods can be used by implementing the `amisc.compression.Compression` base class.

In order to make use of this field quantity when building surrogates, we'll need to call `compression.compute_map()`, which for `SVD` requires passing a data matrix and a desired `rank` of the truncation.

!!! Example "SVD compression"
    To use the SVD compression method, we need to form a "data matrix" of shape `(dof, num_samples)`, where `dof` is the original `(N, Q)` field quantity flattened to `dof`, and `num_samples` are several samples of the full field quantity (such as for varying simulation inputs). In other words, each column of the data matrix is a "snapshot" of the simulation output for this field quantity.

    ```python
    sim_coords = np.random.rand(1000, 3)  # (i.e. load actual Cartesian coords from a result file)
    num_samples = 200
    dof = 3000
    data_matrix = np.empty((dof, num_samples))

    for i in range(num_samples):
        simulation_data = np.random.rand(1000, 3)  # (N, Q) simulation data (i.e. load from a result file)
        data_matrix[:, i] = np.ravel(simulation_data)

    vel = Variable(compression=dict(coords=sim_coords, fields=['ux', 'uy', 'uz'], method='svd'))
    vel.compression.compute_map(data_matrix, rank=10)

    # Now we can use the compression map to compress/reconstruct new values
    new_sim_data  = {'ux': np.random.rand(1000), 'uy': ..., 'uz': ...}
    latent_data   = vel.compress(new_sim_data)
    reconstructed = vel.reconstruct(latent_data)
    ```

Once the compression map has been computed, we can compress or reconstruct new field quantity data:
```python
new_sim_data  = {'field1': ..., 'field2': ...}  # arrays of shape (num_pts,) for each QoI in compression.fields
latent_data   = vel.compress(new_sim_data)      # a single array of shape (rank,) with the key 'latent'
reconstructed = vel.reconstruct(latent_data)    # arrays of shape (num_pts,) for each reconstructed QoI
```

You can optionally pass new coordinates to `compress()` and `reconstruct()`, so that the data will be interpolated to/from any set of coordinates to the original `compression.coords` (e.g. if the new data is not defined on the same grid).

If you also pass a `norm` method to a field quantity `Variable`, then raw simulation data will be normalized first by the indicated method before compression. In general, the compression workflow is _interpolate_ &rarr; _normalize_ &rarr; _compress_ and vice versa for reconstruction. The interpolate step is required to make sure the data aligns with the compression map's coordinates. See `Variable.compress` for more details.

Unlike scalar variables, the domain of a field quantity `Variable` should be a list of domains, one for each "latent" dimension. Since it's typically not practical to know these domains ahead of time, you can either:

1. Use the `Variable` to compress some example data and extract the latent domains manually,
1. Use the built-in `Compression.estimate_latent_ranges()` function (which for `SVD` will compress the `data_matrix` and estimate latent ranges from there),
1. Specify a single, conservative domain (like `(-10, 10)`) that will be used for all the latent dimensions at runtime, or
1. Leave the domain empty, and have `System.fit()` estimate and update the domains from a test set.

The only time you would need to worry about specifying the latent domains is if you are intending on using a field quantity as an input to any of your component models.

As a final note on field quantities, once you've defined and computed the compression map, `amisc` will internally use the compression map during training or prediction to convert the field quantity to/from the latent space. If you have a field quantity named `"vel"` for example, `amisc` will generate latent coefficients with the names `"vel_LATENT0" ... "vel_LATENT1"` and so on up to the total size of the latent space. These temporary latent coefficients will be used as inputs and outputs until they are converted back to the full field quantity. So if you ever inspect raw data arrays returned by `amisc`, you may find these temporary latent coefficients floating around. See the `amisc.to_model_dataset` utility function for reconstructing such arrays.

!!! Note "Object arrays for field quantities"
    If you call `System.predict(use_model=...)` and inspect the true model return values, you will find that field quantities get stored in `numpy` arrays with `dtype=object`. The shape of the object arrays will match the "loop" dimension of the inputs, which is also the same shape as all scalar return values. Each element of the object array will be the field quantity array corresponding to the given input. This allows the field quantity to possibly take on different shapes for different inputs, such as if the model computes a new mesh for each set of inputs.

    For example:
    ```python
    inputs = {'x': np.random.rand(100)}
    outputs = System.predict(inputs, use_model='best')
    scalar = outputs['y']       # numeric array of shape (100,)
    field = outputs['y_field']  # object array of shape  (100,)

    field[0].shape              # (20, 20) for example
    field[1].shape              # (19, 25) for example, if the mesh changed between inputs
    ```

    Your component models should generally return the grid coordinates for field quantities in a special variable name suffixed by `"_coords"`. For example, if your model returns a field quantity named `u_ion`, you would also return the grid coordinates as `u_ion_coords`.
