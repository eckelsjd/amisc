"""Provides serialization protocols for objects in the package. Serialization in the context of `amisc`
means converting an object to a built-in Python object (e.g. string, dictionary, float, etc.). The serialized objects
are then easy to convert to binary or text forms for storage or transmission using various protocols (i.e. pickle,
json, yaml, etc.).

Includes:

- `Serializable` — mixin interface for serializing and deserializing objects
- `Base64Serializable` — mixin class for serializing objects using base64 encoding
- `StringSerializable` — mixin class for serializing objects using string representation
- `PickleSerializable` — mixin class for serializing objects using pickle files
- `YamlSerializable` — metaclass for serializing an object using Yaml load/dump from string
"""
from __future__ import annotations

import base64
import pickle
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from amisc.utils import parse_function_string

__all__ = ['Serializable', 'Base64Serializable', 'StringSerializable', 'PickleSerializable', 'YamlSerializable']

_builtin = str | dict | list | int | float | tuple | bool  # Generic type for common built-in Python objects


class Serializable(ABC):
    """Mixin interface for serializing and deserializing objects."""

    @abstractmethod
    def serialize(self) -> _builtin:
        """Serialize to a builtin Python object."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized_data: _builtin) -> Serializable:
        """Construct a `Serializable` object from serialized data.

        !!! Note "Passing arguments to deserialize"
            Subclasses should generally not take arguments for deserialization. The serialized object should contain
            all the information it needs to reconstruct itself. If you need arguments for deserialization, then
            serialize them along with the object itself and unpack them during the call to deserialize.
        """
        raise NotImplementedError


class Base64Serializable(Serializable):
    """Mixin class for serializing objects using base64 encoding."""
    def serialize(self) -> str:
        return base64.b64encode(pickle.dumps(self)).decode('utf-8')

    @classmethod
    def deserialize(cls, serialized_data: str) -> Base64Serializable:
        return pickle.loads(base64.b64decode(serialized_data))


class StringSerializable(Serializable):
    """Mixin class for serializing objects using string representation."""

    def serialize(self) -> str:
        return str(self)

    @classmethod
    def deserialize(cls, serialized_data: str, trust: bool = False) -> StringSerializable:
        """Deserialize a string representation of the object.

        !!! Warning "Security Risk"
            Only use `trust=True` if you trust the source of the serialized data. This provides a more flexible
            option for `eval`-ing the serialized data from string. By default, this will instead try to parse the
            string as a class signature like `MyClass(*args, **kwargs)`.

        :param serialized_data: the string representation of the object
        :param trust: whether to trust the source of the serialized data (i.e. for `eval`)
        """
        if trust:
            return eval(serialized_data)
        else:
            try:
                name, args, kwargs = parse_function_string(serialized_data)
                return cls(*args, **kwargs)
            except Exception as e:
                raise ValueError(f'String "{serialized_data}" is not a valid class signature.') from e


class PickleSerializable(Serializable):
    """Mixin class for serializing objects using pickle."""
    def serialize(self, save_path: str | Path = None) -> str:
        if save_path is None:
            raise ValueError('Must provide a save path for Pickle serialization.')
        with open(Path(save_path), 'wb') as fd:
            pickle.dump(self, fd)
        return str(Path(save_path).resolve().as_posix())

    @classmethod
    def deserialize(cls, serialized_data: str | Path) -> PickleSerializable:
        with open(Path(serialized_data), 'rb') as fd:
            return pickle.load(fd)


@dataclass
class YamlSerializable(Serializable):
    """Mixin for serializing an object using Yaml load/dump from string."""
    obj: Any

    def serialize(self) -> str:
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', suffix='.yml') as f:
            yaml.dump(self.obj, f, allow_unicode=True)
            f.seek(0)
            s = f.read().strip()
        return s

    @classmethod
    def deserialize(cls, yaml_str: str) -> YamlSerializable:
        obj = yaml.load(yaml_str, yaml.Loader)
        return YamlSerializable(obj=obj)
