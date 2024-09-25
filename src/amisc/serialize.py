"""Provides serialization protocols for objects in the package.

Includes:

- `Serializable` — mixin interface for serializing and deserializing objects
- `Base64Serializable` — mixin class for serializing objects using base64 encoding
- `StringSerializable` — mixin class for serializing objects using string representation
- `PickleSerializable` — mixin class for serializing objects using pickle files
- `MetaSerializable` — metaclass for serializing a `Serializable` type, always with base64 encoding
"""
from __future__ import annotations

import base64
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from amisc.utils import parse_function_string

builtin = str | dict | list | int | float | tuple | bool


class Serializable(ABC):
    """Mixin interface for serializing and deserializing objects."""

    @abstractmethod
    def serialize(self) -> builtin:
        """Serialize to a builtin Python object."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized_data: builtin) -> Serializable:
        """Construct a `Serializable` object from serialized data."""
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
        with open(Path(save_path), 'wb') as fd:
            pickle.dump(self, fd)
        return str(Path(save_path).resolve().as_posix())

    @classmethod
    def deserialize(cls, serialized_data: str | Path) -> PickleSerializable:
        with open(Path(serialized_data), 'rb') as fd:
            return pickle.load(fd)


@dataclass
class MetaSerializable(Base64Serializable):
    """Metaclass for serializing a `Serializable` type, always with base64 encoding."""
    serializer: Type[Serializable]

    def __str__(self):
        return f'Meta({self.serializer.__name__})'
