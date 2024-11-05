"""Efficient framework for building surrogates of multidisciplinary systems using the adaptive multi-index stochastic
collocation (AMISC) technique.

- Author - Joshua Eckels (eckelsjd@umich.edu)
- License - GPL-3.0

The `amisc` package takes an object-oriented approach to building a surrogate of a multidisciplinary system. From the
bottom up, you have:

- **variables** that serve as inputs and outputs for the models,
- **interpolators** that define a specific input &rarr; output mathematical relationship to approximate a function,
- **components** that wrap a model for a single discipline, and a
- **system** that defines the connections between components in a multidisciplinary system.

The **system** is ultimately independent of the specific models, interpolation methods, or underlying variables.
As such, the primary top-level object that users of the `amisc` package will interact with is the `System`.

Variables additionally use `Transform`, `Distribution`, and `Compression` interfaces to manage normalization, PDFs,
and field quantity compression, respectively.

Currently, only Lagrange polynomial interpolation is implemented as the underlying surrogate method with a
sparse grid data structure. SVD is also the only currently implemented method for compression. However, interfaces
are provided for `Interpolator`, `TrainingData`, and `Compression` to allow for easy extension to other methods.

Here is a class diagram summary of this workflow:

``` mermaid
classDiagram
    namespace Core {
        class System {
          +list[Component] components
          +TrainHistory train_history
          +fit()
          +predict(x)
        }
        class Component {
          +callable model
          +list[Variable] inputs
          +list[Variable] outputs
          +Interpolator interpolator
          +TrainingData training_data
          +activate_index(alpha, beta)
          +predict(x)
        }
        class Variable {
          +str name
          +tuple domain
          +Distribution dist
          +Transform norm
          +Compression comp
          +sample()
          +normalize()
          +compress()
        }
    }
    class Interpolator {
      <<abstract>>
      + refine()
      + predict(x)
    }
    class TrainingData {
      <<abstract>>
      +get()
      +set()
      +refine()
    }
    class Transform {
      <<abstract>>
      +transform(values)
    }
    class Distribution {
      <<abstract>>
      +sample(size)
      +pdf(x)
    }
    class Compression {
      <<abstract>>
      +compress(values)
      +reconstruct(values)
    }
    System --o "1..n" Component
    Component --o "1..n" Variable
    direction TD
    Component --* Interpolator
    Component --* TrainingData
    Variable --o Transform
    Variable --o Distribution
    Variable --o Compression
```

Note how the `System` aggregates the `Component`, which aggregates the `Variable`. In other
words, variables can act independently of components, and components can act independently of systems. Components
make use of `Interpolator` and `TrainingData` interfaces to manage the underlying surrogate method
and training data, respectively. Similarly, `Variables` use `Transform`, `Distribution`, and `Compression`
interfaces to manage normalization, PDFs, and field quantity compression, respectively.

The `amisc` package also includes a `FileLoader` interface for loading and dumping `amisc` objects to/from file.
We recommend using the built-in `YamlLoader` for this purpose, as it includes custom YAML tags for reading/writing
`amisc` objects from file.
"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from pathlib import Path as _Path
from typing import Any as _Any

import yaml as _yaml

from amisc.component import Component
from amisc.system import System
from amisc.utils import to_model_dataset, to_surrogate_dataset
from amisc.variable import Variable, VariableList

__version__ = "0.5.1"
__all__ = ['System', 'Component', 'Variable', 'VariableList', 'FileLoader', 'YamlLoader',
           'to_model_dataset', 'to_surrogate_dataset']


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
        except (TypeError, OSError, FileNotFoundError):
            return _yaml.load(stream, Loader=cls._yaml_loader())

    @classmethod
    def dump(cls, obj, stream):
        try:
            with open(_Path(stream).with_suffix('.yml'), 'w', encoding='utf-8') as fd:
                return _yaml.dump(obj, fd, Dumper=cls._yaml_dumper(), allow_unicode=True, sort_keys=False)
        except (TypeError, OSError, FileNotFoundError):
            return _yaml.dump(obj, stream, Dumper=cls._yaml_dumper(), allow_unicode=True, sort_keys=False)
