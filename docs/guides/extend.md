This guide covers how to extend `amisc` for implementing custom surrogate methods, including compression methods, PDFs, and interpolation. We will walk through each of the 7 abstract interfaces that may be implemented by the user:

- [Transform][amisc.transform.Transform] - methods for variable normalization,
- [Distribution][amisc.distribution.Distribution] - methods for probability distribution functions,
- [Compression][amisc.compression.Compression] - methods for field quantity compression,
- [TrainingData][amisc.training.TrainingData] - methods for training data selection (i.e. experimental design) and storage,
- [Interpolator][amisc.interpolator.Interpolator] - methods for approximating the input &rarr; output mapping of a model,
- [ModelKwargs][amisc.component.ModelKwargs] - dataclass for passing extra arguments to the component models.
- [FileLoader][amisc.FileLoader] - methods for loading `amisc` objects to/from file.

The class layout of these interfaces is summarized in the [API reference](../reference/index.md).

!!! Note "Important"
    All custom objects implementing any of the interfaces above must also implement the `serialize` and `deserialize` mixin methods to allow saving and loading the custom objects from file. See the [serialization](#serialization) section.

## Transform
Transforms provide a method for normalizing data for surrogate training. `Transform` objects must implement the `transform` method:
```python
from amisc.transform import Transform

class CustomTransform(Transform):
    """A transform that adds 1."""

    def _transform(self, values, inverse=False):
        return values - 1 if inverse else values + 1
```

The `transform` method should additionally handle the `inverse` flag to denormalize the provided values. Currently available transforms include log, linear, minmax, and z-score.

## Distribution
Distributions define a probability distribution function. `Distribution` objects should implement the `sample` and `pdf` methods:
```python
from amisc.distribution import Distribution

class CustomDistribution(Distribution):
    """A custom distribution that returns a constant value."""

    def sample(self, size=1):
        """Generate samples from the distribution."""
        return np.full(size, 42)

    def pdf(self, x):
        """Probability density function of the distribution at `x` locations."""
        return np.ones_like(x)
```

Currently available distributions include uniform, normal, log-uniform, and log-normal.

## Compression
Compression methods are responsible for compressing and reconstructing high-dimensional data to/from a compressed or "latent" space. Along with implementing the `compress` and `reconstruct` methods, `Compression` objects should keep track of the latent space size and implement a `compute_map` method, which generates the map to/from the latent space. The `Compression` object can also optionally provide a method for estimating the domain of the latent space.
```python
from amisc.compression import Compression

class CustomCompression(Compression):
    """A custom compression method."""

    def compute_map(self, **kwargs):
        """Compute and store the compression map."""
        self.coords = kwargs.get('coords', None)
        self._is_computed = True

    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress the data into a latent space."""
        return data / 2  # Example compression

    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        """Reconstruct the compressed data back into the full `dof` space."""
        return compressed * 2  # Example reconstruction

    def latent_size(self) -> int:
        """Return the size of the latent space."""
        return 10  # Example latent size

    def estimate_latent_ranges(self) -> list[tuple[float, float]]:
        """Estimate the range of the latent space coefficients."""
        return [(0.0, 1.0) for _ in range(self.latent_size())]
```

Only SVD compression is currently available.

## Training data
The `TrainingData` interface allows users to implement custom methods for training data selection, storage, and retrieval. This interface defines how training data is managed within the framework, including experimental design and data fidelity. Data storage and retrieval is generally guided by the $(\alpha, \beta)$ fidelity indices. See the [model fidelity](interface_models.md) docs for more details.

Custom training data methods should implement the `TrainingData` interface:
```python
from amisc.training import TrainingData

class CustomTrainingData(TrainingData):
    """A custom training data method."""

    def refine(self, alpha, beta, domains, weight_fcns):
        """Refine the training data based on the given (alpha, beta) fidelity and domains."""
        # Implement refinement (experimental design) logic here
        pass

    def set(self, alpha, beta, idx, data):
        """Set the training data for the given indices and (alpha, beta) fidelity."""
        # Implement data storage logic here
        pass

    def get(self, alpha, beta):
        """Retrieve training data by (alpha, beta) fidelity."""
        # Implement data retrieval logic here
        pass

    def clear(self):
        """Clear all data."""
        pass

    def set_errors(self, alpha, beta, idx, errors):
        """Optionally store error information for failed model evaluations."""
        pass

    def impute_missing_data(self, alpha, beta):
        """Optionally impute training data for failed model evaluations."""
        pass
```

Currently, the only available training data method is the `SparseGrid`. This method stores data by its location in a larger tensor-product grid of the input dimensions, which interfaces naturally with the MISC framework. The experimental design is guided by the space-filling Leja objective function. See [SparseGrid][amisc.training.SparseGrid] for more details.

## Interpolator
Interpolators approximate the input &rarr; output mapping of a model given a set of training data. `Interpolator` objects must implement the `refine` and `predict` methods:

```python
from amisc.interpolator import Interpolator, InterpolatorState

class CustomInterpolator(Interpolator):
    """A custom interpolator."""

    def refine(self, beta, training_data, old_state, domains) -> InterpolatorState:
        """Refine the old interpolator state with new training data for a given beta fidelity index."""
        # Implement refinement logic here
        pass

    def predict(self, x, state, training_data):
        """Predict the output for given input `x` using the interpolator state and training data."""
        # Implement prediction logic here
        pass
```

The `predict` method computes the interpolator approximation at a new set of points `x` given an interpolator state and set of training data. The `refine` method takes new training data and updates the old interpolator state.

The `InterpolatorState` is an interface for storing the internal state (e.g. weights and biases) of the interpolator method, and is used for computing the interpolator approximation along with a set of training data. If you implement the `Interpolator` interface, you should also implement a corresponding `InterpolatorState` that integrates with your custom interpolator:
```python
from amisc.interpolator import InterpolatorState

class CustomInterpolatorState(InterpolatorState):
    """The internal state for a custom interpolator."""
    # Define additional state variables here
    pass
```

Currently, the available methods are `Lagrange` polynomial interpolation and `Linear` regression. The state of a `Lagrange` polynomial includes the 1d grids and barycentric weights for each input dimension. The state of a `Linear` regression includes the underlying [scikit-learn](https://scikit-learn.org/stable/) linear model. See [Lagrange][amisc.interpolator.Lagrange] and [Linear][amisc.interpolator.Linear] for more details. Note that linear regression also includes options for polynomial features.

## Model keyword arguments
The `ModelKwarg` interface provides a dataclass for passing extra options to the underlying component models. The default is a simple `dict` that gets passed as a set of `key=value` pairs. The primary reason for overriding this class is if you have complicated arguments that require custom serialization. See the [serialization](#serialization) section below.

## Serialization
The [Serializable][amisc.serialize.Serializable] interface defines the `serialize` and `deserialize` methods for converting `amisc` objects to/from built-in Python types (such as `strings`, `floats`, and `dicts`).

!!! Note "Important"
    All custom objects implementing any of the interfaces above must also implement the `serialize` and `deserialize` mixin methods to allow saving and loading the custom objects from file.

Note that the `Variable`, `Component`, and `System` classes themselves implement the `Serializable` interface, so that they too can be easily saved and loaded from file. In the context of `amisc`, "serialization" means "convert to built-in Python types" (and vice versa for deserialization). This typically means converting objects to dictionaries with `key=value` pairs for all of the object's properties. Several serialization methods are provided in the [`amisc.serialize`][amisc.serialize] module. These can be mixed-in to your custom objects by inheritance.

!!! Example "Mixin serialization methods"
    Our custom interpolator state can be serialized to [base64 encoding](https://docs.python.org/3/library/base64.html) by inheriting from the [Base64Serializable][amisc.serialize.Base64Serializable] mixin class:
    ```python
    class CustomInterpolatorState(InterpolatorState, Base64Serializable):
        """The internal state of a custom interpolator."""
        # Define additional state variables here
        pass
    ```

## FileLoader
The `FileLoader` interface defines the `dump` and `load` methods for saving and loading `amisc` objects to/from file. The recommended way to save objects to file is the provided [YamlLoader][amisc.YamlLoader] class, which uses the registered `!Variable` (etc.) tags to construct `amisc` objects directly from `.yml` config files.

However, more generally, you may define a custom `FileLoader` if you prefer to work with different file types. For example, you can make use of the fact that all `amisc` objects are [serializable](#serialization) to/from built-in Python types to more easily transcribe objects to a new file format (built-in Python types like `strings` and `dicts` readily convert to most file types).

!!! Example "A custom JSON file loader"
    This example shows saving and loading a [System][amisc.system.System] object from a JSON file (using the `serialize` and `deserialize` methods). More generally, you would also want to handle `Variables` and `Components`.
    ```python
    from amisc import FileLoader
    import json

    class JSONLoader(FileLoader):
        """Save and load amisc objects from JSON"""

        def load(self, file)
            """Load an amisc.System object (for example)"""
            with open(file, 'r') as fd:
                data = json.load(fd)
            return System.deserialize(data)

        def dump(self, obj, file):
            """Dump an amisc.System object (for example)"""
            with open(file, 'w') as fd:
                json.dump(obj.serialize(), fd)
    ```
