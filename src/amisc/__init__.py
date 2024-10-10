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

Variables additionally use `Transform` and `Distribution` objects to manage normalization and PDFs, respectively.

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
      +str name
      +tuple domain
      +str units
      +Distribution dist
      +Transform norm
      +float nominal
      +normalize()
      +compress()
    }
    class Transform {
      +transform(values)
    }
    class Distribution {
      +sample(size)
      +pdf(x)
    }
    SystemSurrogate o-- "1..n" ComponentSurrogate
    ComponentSurrogate o-- "1..n" BaseInterpolator
    direction LR
    ComponentSurrogate <|-- SparseGridSurrogate
    BaseInterpolator <|-- LagrangeInterpolator
    SparseGridSurrogate ..> LagrangeInterpolator
    Variable o-- Transform
    Variable o-- Distribution
```
Note how the `SystemSurrogate` aggregates the `ComponentSurrogate`, which aggregates the `BaseInterpolator`. In other
words, interpolators can act independently of components, and components can act independently of systems. All three
make use of the `Variable` objects (these connections are not shown for visual clarity). Currently, the only
underlying surrogate method that is implemented here is Lagrange polynomial interpolation (i.e. the
`LagrangeInterpolator`). If one wanted to use neural networks instead, the only change required is a new
implementation of `BaseInterpolator`.
"""
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from pathlib import Path as _Path
from typing import Any as _Any

import yaml as _yaml

from amisc.variable import Variable, VariableList
from amisc.component import Component
from amisc.system import System

__version__ = "0.4.0"
__all__ = ['System', 'Component', 'Variable', 'VariableList', 'FileLoader', 'YamlLoader']


class FileLoader(_ABC):
    """Common interface for loading and dumping `amisc` objects to/from file."""

    @classmethod
    @_abstractmethod
    def load(cls, stream: str | _Path | _Any):
        """Load an `amisc` object from a stream. If a file path is given, will attempt to open the file."""
        raise NotImplementedError

    @classmethod
    @_abstractmethod
    def dump(cls, obj, stream: str | _Path | _Any):
        """Save an `amisc` object to a stream. If a file path is given, will attempt to write to the file."""
        raise NotImplementedError


class YamlLoader(FileLoader):
    """YAML file loader for `amisc` objects."""

    @staticmethod
    def _yaml_loader():
        """Custom YAML loader that includes `amisc` object tags."""
        loader = _yaml.Loader
        loader.add_constructor(Variable.yaml_tag, Variable._yaml_constructor)
        loader.add_constructor(VariableList.yaml_tag, VariableList._yaml_constructor)
        loader.add_constructor(Component.yaml_tag, Component._yaml_constructor)
        loader.add_constructor(System.yaml_tag, System._yaml_constructor)
        return loader

    @staticmethod
    def _yaml_dumper():
        """Custom YAML dumper that includes `amisc` object tags."""
        dumper = _yaml.Dumper
        dumper.add_representer(Variable, Variable._yaml_representer)
        dumper.add_representer(VariableList, VariableList._yaml_representer)
        dumper.add_representer(Component, Component._yaml_representer)
        dumper.add_representer(System, System._yaml_representer)
        return dumper

    @classmethod
    def load(cls, stream):
        try:
            with open(_Path(stream).with_suffix('.yml'), 'r', encoding='utf-8') as fd:
                return _yaml.load(fd, Loader=cls._yaml_loader())
        except:
            return _yaml.load(stream, Loader=cls._yaml_loader())

    @classmethod
    def dump(cls, obj, stream):
        try:
            with open(_Path(stream).with_suffix('.yml'), 'w', encoding='utf-8') as fd:
                return _yaml.dump(obj, fd, Dumper=cls._yaml_dumper(), allow_unicode=True, sort_keys=False)
        except:
            return _yaml.dump(obj, stream, Dumper=cls._yaml_dumper(), allow_unicode=True, sort_keys=False)
