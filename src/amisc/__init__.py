"""Efficient framework for building surrogates of multidisciplinary systems using the adaptive multi-index stochastic
collocation (AMISC) technique.

- Author - Joshua Eckels (eckelsjd@umich.edu)
- License - GPL-3.0

The `amisc` package takes an object-oriented approach to building a surrogate of a multidisciplinary system. From the
bottom up, you have:

- **variables** that serve as inputs and outputs for the models,
- **interpolators** that define a specific input &rarr; output mathematical relationship to interpolate a function,
- **components** that wrap a model for a single discipline, and a
- **system** that defines the connections between components in a multidisciplinary system.

The variables, interpolators, and components all have abstract base classes, so that the **system** is ultimately
independent of the specific models, interpolation methods, or underlying variables. As such, the primary top-level
object that users of the `amisc` package will interact with is the `SystemSurrogate`.

!!! Note
    There are already pretty good implementations of the other abstractions that most users will not need to worry
    about, but they are provided in this API reference for completeness. The abstractions allow new interpolation
    (i.e. function approximation) methods to be implemented if desired, such as neural networks, kriging, etc.

Here is a class diagram summary of this workflow:

``` mermaid
classDiagram
    namespace Core {
        class SystemSurrogate {
          +list[Variable] exo_vars
          +list[Variable] coupling_vars
          +int refine_level
          +fit()
          +predict(x)
          +sample_inputs(size)
          +insert_component(comp)
        }
        class ComponentSurrogate {
          <<abstract>>
          +IndexSet index_set
          +IndexSet candidate_set
          +list[Variable] x_vars
          +dict[str: BaseInterpolator] surrogates
          +dict[str: float] misc_coeff
          +predict(x)
          +activate_index(alpha, beta)
          +add_surrogate(alpha, beta)
          +update_misc_coeff()
        }
        class BaseInterpolator {
          <<abstract>>
          +tuple beta
          +list[Variable] x_vars
          +np.ndarray xi
          +np.ndarray yi
          +set_yi()
          +refine()
          +__call__(x)
        }
    }
    class SparseGridSurrogate {
      +np.ndarray x_grids
      +dict xi_map
      +dict yi_map
      +get_tensor_grid(alpha, beta)
    }
    class LagrangeInterpolator {
      +np.ndarray x_grids
      +np.ndarray weights
      +get_grid_sizes()
      +leja_1d()
    }
    class Variable {
      +str var_id
      +tuple domain
      +str units
      +str dist
      +float nominal
      +pdf(x)
      +sample(size)
      +normalize()
      +compress()
    }
    SystemSurrogate o-- "1..n" ComponentSurrogate
    ComponentSurrogate o-- "1..n" BaseInterpolator
    direction LR
    ComponentSurrogate <|-- SparseGridSurrogate
    BaseInterpolator <|-- LagrangeInterpolator
    SparseGridSurrogate ..> LagrangeInterpolator
    Variable
```
Note how the `SystemSurrogate` aggregates the `ComponentSurrogate`, which aggregates the `BaseInterpolator`. In other
words, interpolators can act independently of components, and components can act independently of systems. All three
make use of the `Variable` objects (these connections are not shown for visual clarity). Currently, the only
underlying surrogate method that is implemented here is Lagrange polynomial interpolation (i.e. the
`LagrangeInterpolator`). If one wanted to use neural networks instead, the only change required is a new
implementation of `BaseInterpolator`.
"""
import numpy as np
import yaml

from amisc.interpolator import BaseInterpolator
from amisc.variable import Variable

__version__ = "0.3.0"

yaml.add_representer(Variable, Variable._yaml_representer)
yaml.add_constructor(Variable.yaml_tag, Variable._yaml_constructor)

# Custom types that are used frequently
IndexSet = list[tuple[tuple, tuple]]
MiscTree = dict[str: dict[str: float | BaseInterpolator]]
InterpResults = BaseInterpolator | tuple[list[int | tuple | str], np.ndarray, BaseInterpolator]
IndicesRV = list[int | str | Variable] | int | str | Variable
