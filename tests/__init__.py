"""Tests for the `amisc` package. Ordered according to the following hierarchy of modules, with higher-level
modules dependent on the lower-levels.

Testing modules:

- `test_utils` - tests for the `amisc.utils` module including dataset conversion, formatting, and other utilities.
- `test_serialize` - tests for the `amisc.serialize` module including serialization mixin classes.
- `test_variable` - tests for the `amisc.variable` module including the `Variable` and `VariableList` classes,
                    as well as `Transform`, `Distribution`, and `Compression` interfaces.
- `test_training_data` - tests for the `amisc.training` module including the `SparseGrid` implementation.
- `test_interpolator` - tests for the `amisc.interpolator` module including the `Lagrange` implementation.
- `test_component` - tests for the `amisc.component` module including the `Component` model wrapper and its
                     associated data structures `MiscTree` and `IndexSet`.
- `test_system_spec` - tests for the `amisc.system` module, primarily focused on pydantic validation and serialization.
- `test_system` - tests for the `amisc.system` module, primarily focused on the `System` class and its methods.
- `test_convergence` - tests for the `amisc.system` module to ensure convergence of the `System` surrogate on
                       various test models.

Profiling modules:

- `profile_fit` - profile the performance of `System.fit` for various configurations.
- `profile_predict` - profile the performance of `System.predict` using `py-spy`.
"""