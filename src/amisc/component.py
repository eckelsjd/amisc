"""A `Component` is an `amisc` wrapper around a single discipline model. It manages surrogate construction and
a hierarchy of modeling fidelities.

Includes:

- `ModelKwargs` — a dataclass for storing model keyword arguments
- `StringKwargs` — a dataclass for storing model keyword arguments as a string
- `IndexSet` — a dataclass that maintains a list of multi-indices
- `MiscTree` — a dataclass that maintains MISC data in a `dict` tree, indexed by `alpha` and `beta`
- `SurrogateStatus` — an enumeration that keeps track of what state MISC coefficients are in for surrogate evaluation
- `Component` — a dataclass that manages a single discipline model and its surrogate hierarchy
"""
from __future__ import annotations

import ast
import copy
import inspect
import itertools
import logging
import os
import random
import string
import tempfile
import typing
import warnings
from abc import ABC
from collections import UserDict, deque
from concurrent.futures import ALL_COMPLETED, Executor, wait
from enum import IntFlag
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Literal, Optional

import numpy as np
import yaml
from joblib import delayed
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from typing_extensions import TypedDict

from amisc.interpolator import InterpolatorState, Interpolator, Lagrange
from amisc.serialize import YamlSerializable, PickleSerializable, Serializable, StringSerializable
from amisc.training import TrainingData, SparseGrid
from amisc.typing import MultiIndex, Dataset
from amisc.utils import get_logger, format_inputs, format_outputs, search_for_file
from amisc.utils import _get_yaml_path, _inspect_assignment, _inspect_function
from amisc.variable import Variable, VariableList

__all__ = ["ModelKwargs", "StringKwargs", "SurrogateStatus", "IndexSet", "MiscTree", "Component"]
_VariableLike = list[Variable | dict | str] | str | Variable | dict | VariableList  # Generic type for Variables


class ModelKwargs(UserDict, Serializable):
    """Default dataclass for storing model keyword arguments."""

    def serialize(self):
        return self.data

    @classmethod
    def deserialize(cls, serialized_data):
        return ModelKwargs(**serialized_data)

    @classmethod
    def from_dict(cls, config: dict) -> ModelKwargs:
        """Create a `ModelKwargs` object from a `dict` configuration."""
        method = config.pop('method', 'default_kwargs').lower()
        match method:
            case 'default_kwargs':
                return ModelKwargs(**config)
            case 'string_kwargs':
                return StringKwargs(**config)
            case other:
                config['method'] = other
                return ModelKwargs(**config)  # Pass the method through


class StringKwargs(StringSerializable, ModelKwargs):
    """Dataclass for storing model keyword arguments as a string."""
    def __repr__(self):
        return str(self.data)

    def __str__(self):
        def format_value(value):
            if isinstance(value, str):
                return f'"{value}"'
            else:
                return str(value)

        kw_str = ", ".join([f"{key}={format_value(value)}" for key, value in self.items()])
        return f"ModelKwargs({kw_str})"


class IndexSet(set, Serializable):
    """Dataclass that maintains a list of multi-indices. Overrides basic `set` functionality to ensure
    elements are formatted correctly as `(alpha, beta)`.

    !!! Example "An example index set"
        $\\mathcal{I} = [(\\alpha, \\beta)_1 , (\\alpha, \\beta)_2, (\\alpha, \\beta)_3 , ...]$ would be specified
        as `I = [((0, 0), (0, 0, 0)) , ((0, 1), (0, 1, 0)), ...]`.
    """
    def __init__(self, s=()):
        s = [self._validate_element(ele) for ele in s]
        super().__init__(s)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return self.__str__()

    def add(self, __element):
        super().add(self._validate_element(__element))

    def update(self, __elements):
        super().update([self._validate_element(ele) for ele in __elements])

    @classmethod
    def _validate_element(cls, element):
        """Validate that the element is a tuple of two multi-indices."""
        alpha, beta = ast.literal_eval(element) if isinstance(element, str) else tuple(element)
        return MultiIndex(alpha), MultiIndex(beta)

    @classmethod
    def _wrap_methods(cls, names):
        """Make sure set operations return an `IndexSet` object."""
        def wrap_method_closure(name):
            def inner(self, *args):
                result = getattr(super(cls, self), name)(*args)
                if isinstance(result, set):
                    result = cls(result)
                return result
            inner.fn_name = name
            setattr(cls, name, inner)

        for name in names:
            wrap_method_closure(name)

    def serialize(self) -> list[str]:
        """Return a list of each multi-index in the set serialized to a string."""
        return [str(ele) for ele in self]

    @classmethod
    def deserialize(cls, serialized_data: list[str]) -> IndexSet:
        """Deserialize using pydantic model validation on `IndexSet.data`."""
        return cls(serialized_data)


IndexSet._wrap_methods(['__ror__', 'difference_update', '__isub__', 'symmetric_difference', '__rsub__', '__and__',
                        '__rand__', 'intersection', 'difference', '__iand__', 'union', '__ixor__',
                        'symmetric_difference_update', '__or__', 'copy', '__rxor__', 'intersection_update', '__xor__',
                        '__ior__', '__sub__'
                        ])


class MiscTree(UserDict, Serializable):
    """Dataclass that maintains MISC data in a `dict` tree, indexed by `alpha` and `beta`. Overrides
    basic `dict` functionality to ensure elements are formatted correctly as `(alpha, beta) -> data`.
    Used to store MISC coefficients, model costs, and interpolator states.

    The underlying data structure is: `dict[MultiIndex, dict[MultiIndex, float | InterpolatorState]]`
    """
    def __init__(self, data: dict = None, **kwargs):
        data_dict = data or {}
        if isinstance(data_dict, MiscTree):
            data_dict = data_dict.data
        data_dict.update(kwargs)
        super().__init__(self._validate_data(data_dict))

    def serialize(self, *args, keep_yaml_objects=False, **kwargs) -> dict:
        """Serialize `alpha, beta` indices to string and return a `dict` of internal data.

        :param args: extra serialization arguments for internal `InterpolatorState`
        :param keep_yaml_objects: whether to keep `YamlSerializable` instances in the serialization
        :param kwargs: extra serialization keyword arguments for internal `InterpolatorState`
        """
        ret_dict = {}
        if state_serializer := self.state_serializer(self.data):
            ret_dict['state_serializer'] = state_serializer.obj if keep_yaml_objects else state_serializer.serialize()
        for alpha, beta, data in self:
            ret_dict.setdefault(str(alpha), dict())
            serialized_data = data.serialize(*args, **kwargs) if isinstance(data, InterpolatorState) else float(data)
            ret_dict[str(alpha)][str(beta)] = serialized_data
        return ret_dict

    @classmethod
    def deserialize(cls, serialized_data: dict) -> MiscTree:
        """"Deserialize using pydantic model validation on `MiscTree` data.

        :param serialized_data: the data to deserialize to a `MiscTree` object
        """
        return cls(serialized_data)

    @classmethod
    def state_serializer(cls, data: dict) -> YamlSerializable | None:
        """Infer and return the state serializer from the `MiscTree` data."""
        serializer = data.get('state_serializer', None)  # if `data` is serialized
        if serializer is None:  # Otherwise search for an InterpolatorState
            for alpha, beta_dict in data.items():
                if alpha == 'state_serializer':
                    continue
                for beta, value in beta_dict.items():
                    if isinstance(value, InterpolatorState):
                        serializer = type(value)
                        break
                if serializer is not None:
                    break
        return cls._validate_state_serializer(serializer)

    @classmethod
    def _validate_state_serializer(cls, state_serializer: Optional[str | type[Serializable] | YamlSerializable]
                                   ) -> YamlSerializable | None:
        if state_serializer is None:
            return None
        elif isinstance(state_serializer, YamlSerializable):
            return state_serializer
        elif isinstance(state_serializer, str):
            return YamlSerializable.deserialize(state_serializer)  # Load the serializer type from string
        else:
            return YamlSerializable(obj=state_serializer)

    @classmethod
    def _validate_data(cls, serialized_data: dict) -> dict:
        state_serializer = cls.state_serializer(serialized_data)
        ret_dict = {}
        for alpha, beta_dict in serialized_data.items():
            if alpha == 'state_serializer':
                continue
            alpha_tup = MultiIndex(alpha)
            ret_dict.setdefault(alpha_tup, dict())
            for beta, data in beta_dict.items():
                beta_tup = MultiIndex(beta)
                if isinstance(data, InterpolatorState):
                    pass
                elif state_serializer is not None:
                    data = state_serializer.obj.deserialize(data)
                else:
                    data = float(data)
                assert isinstance(data, InterpolatorState | float)
                ret_dict[alpha_tup][beta_tup] = data
        return ret_dict

    @staticmethod
    def _is_alpha_beta_access(key):
        """Check that the key is of the format `(alpha, beta).`"""
        return (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str | tuple)
                and isinstance(key[1], str | tuple))

    def get(self, key, default=None) -> float | InterpolatorState:
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def update(self, data_dict: dict = None, **kwargs):
        """Force `dict.update()` through the validator."""
        data_dict = data_dict or dict()
        data_dict.update(kwargs)
        super().update(self._validate_data(data_dict))

    def __setitem__(self, key: tuple | MultiIndex, value: float | InterpolatorState):
        """Allows `misc_tree[alpha, beta] = value` usage."""
        if self._is_alpha_beta_access(key):
            alpha, beta = MultiIndex(key[0]), MultiIndex(key[1])
            self.data.setdefault(alpha, dict())
            self.data[alpha][beta] = value
        else:
            super().__setitem__(MultiIndex(key), value)

    def __getitem__(self, key: tuple | MultiIndex) -> float | InterpolatorState:
        """Allows `value = misc_tree[alpha, beta]` usage."""
        if self._is_alpha_beta_access(key):
            alpha, beta = MultiIndex(key[0]), MultiIndex(key[1])
            return self.data[alpha][beta]
        else:
            return super().__getitem__(MultiIndex(key))

    def __eq__(self, other):
        if isinstance(other, MiscTree):
            try:
                for alpha, beta, data in self:
                    if other[alpha, beta] != data:
                        return False
                return True
            except KeyError:
                return False
        else:
            return False

    def __iter__(self) -> Iterable[tuple[tuple, tuple, float | InterpolatorState]]:
        for alpha, beta_dict in self.data.items():
            if alpha == 'state_interpolator':
                continue
            for beta, data in beta_dict.items():
                yield alpha, beta, data


class SurrogateStatus(IntFlag):
    """Keeps track of what state MISC coefficients are in for surrogate evaluation.

    - TRAINING - training mode; only `active_set` indices are used
    - EVALUATION - evaluation mode; `active_set` and `candidate_set` indices are used
    - RESET - reset mode; unknown state, MISC coefficients should be recomputed
    """
    TRAINING = 1
    EVALUATION = 2
    RESET = 3


class ComponentSerializers(TypedDict, total=False):
    """Type hint for the `Component` class data serializers."""
    model_kwargs: str | type[Serializable] | YamlSerializable
    interpolator: str | type[Serializable] | YamlSerializable
    training_data: str | type[Serializable] | YamlSerializable


class Component(BaseModel, Serializable):
    yaml_tag: ClassVar[str] = u'!Component'
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, validate_default=True,
                              protected_namespaces=(), extra='allow')
    # Configuration
    serializers: Optional[ComponentSerializers] = None
    name: Optional[str] = None
    model: str | Callable[[dict | Dataset, ...], dict | Dataset]
    model_kwargs: str | dict | ModelKwargs = {}
    inputs: _VariableLike
    outputs: _VariableLike
    max_alpha: str | tuple = MultiIndex()
    max_beta_train: str | tuple = MultiIndex()
    max_beta_interpolator: str | tuple = MultiIndex()
    interpolator: Any | Interpolator = Lagrange()
    vectorized: bool = False
    call_unpacked: Optional[bool] = None  # If the model expects inputs/outputs like `func(x1, x2, ...)->(y1, y2, ...)
    ret_unpacked: Optional[bool] = None

    # Data storage/states for a MISC component
    active_set: list | set | IndexSet = IndexSet()
    candidate_set: list | set | IndexSet = IndexSet()
    misc_states: dict | MiscTree = MiscTree()          # (alpha, beta) -> Interpolator state
    misc_costs: dict | MiscTree = MiscTree()           # (alpha, beta) -> Added computational cost for this mult-index
    misc_coeff_train: dict | MiscTree = MiscTree()     # (alpha, beta) -> c_[alpha, beta] (active set only)
    misc_coeff_test: dict | MiscTree = MiscTree()      # (alpha, beta) -> c_[alpha, beta] (including candidate set)
    model_costs: dict = dict()  # Average single fidelity model costs (for each alpha)
    training_data: Any | TrainingData = SparseGrid()
    status: int | SurrogateStatus = SurrogateStatus.RESET

    # Internal
    _logger: Optional[logging.Logger] = None
    _executor: Optional[Executor] = None

    def __init__(self, /, model, *args, inputs=None, outputs=None, executor=None, name=None, **kwargs):
        if name is None:
            name = _inspect_assignment('Component')  # try to assign the name from inspection
        name = name or model.__name__ or "Component_" + "".join(random.choices(string.digits, k=3))

        # Determine how the model expects to be called and gather inputs/outputs
        _ = self._validate_model_signature(model, args, inputs, outputs, kwargs.get('call_unpacked', None),
                                           kwargs.get('ret_unpacked', None))
        model, inputs, outputs, call_unpacked, ret_unpacked = _
        kwargs['call_unpacked'] = call_unpacked
        kwargs['ret_unpacked'] = ret_unpacked

        # Gather all model kwargs (anything else passed in for kwargs is assumed to be a model kwarg)
        model_kwargs = kwargs.get('model_kwargs', {})
        for key in kwargs.keys() - self.model_fields.keys():
            model_kwargs[key] = kwargs.pop(key)
        kwargs['model_kwargs'] = model_kwargs

        # Gather data serializers from type checks (if not passed in as a kwarg)
        serializers = kwargs.get('serializers', {})  # directly passing serializers will override type checks
        for key in ComponentSerializers.__annotations__.keys():
            field = kwargs.get(key, None)
            if isinstance(field, dict):
                field_super = next(filter(lambda x: issubclass(x, Serializable),
                                          typing.get_args(self.model_fields[key].annotation)), None)
                field = field_super.from_dict(field) if field_super is not None else field
            if not serializers.get(key, None):
                serializers[key] = type(field) if isinstance(field, Serializable) else (
                    type(self.model_fields[key].default))
        kwargs['serializers'] = serializers

        super().__init__(model=model, inputs=inputs, outputs=outputs, name=name, **kwargs)  # Runs pydantic validation

        # Set internal properties
        assert self.is_downward_closed(self.active_set.union(self.candidate_set))
        self.executor = executor

    @classmethod
    def _validate_model_signature(cls, model, args=(), inputs=None, outputs=None,
                                  call_unpacked=None, ret_unpacked=None):
        """Parse model signature and decide how the model expects to be called based on what input/output information
        is provided or inspected from the model signature.
        """
        if inputs is not None:
            inputs = cls._validate_variables(inputs)
        if outputs is not None:
            outputs = cls._validate_variables(outputs)
        model = cls._validate_model(model)

        # Default to `dict` (i.e. packed) model call/return signatures
        if call_unpacked is None:
            call_unpacked = False
        if ret_unpacked is None:
            ret_unpacked = False
        inputs_inspect, outputs_inspect = _inspect_function(model)
        call_unpacked = call_unpacked or (len(inputs_inspect) > 1)  # Assume multiple inputs require unpacking
        ret_unpacked = ret_unpacked or (len(outputs_inspect) > 1)   # Assume multiple outputs require unpacking

        # Extract inputs/outputs from args
        arg_inputs = ()
        arg_outputs = ()
        if len(args) > 0:
            if call_unpacked:
                if isinstance(args[0], dict | str | Variable):
                    arg_inputs = args[:len(inputs_inspect)]
                    arg_outputs = args[len(inputs_inspect):]
                else:
                    arg_inputs = args[0]
                    arg_outputs = args[1:]
            else:
                arg_inputs = args[0]    # Assume first arg is a single or list of inputs
                arg_outputs = args[1:]  # Assume rest are outputs

        # Resolve inputs
        inputs = inputs or []
        inputs = VariableList.merge(inputs, arg_inputs)
        if len(inputs) == 0:
            inputs = inputs_inspect
            call_unpacked = True
            if len(inputs) == 0:
                raise ValueError("Could not infer input variables from model signature. Either your model does not "
                                 "accept input arguments or an error occurred during inspection.\nPlease provide the "
                                 "inputs directly as `Component(inputs=[...])` or fix the model signature.")
        if call_unpacked:
            if not all([var == inputs_inspect[i] for i, var in enumerate(inputs)]):
                warnings.warn(f"Mismatch between provided inputs: {inputs.values()} and inputs inferred from "
                              f"model signature: {inputs_inspect}. This may cause unexpected results.")
        else:
            if len(inputs_inspect) > 1:
                warnings.warn(f"Model signature expects multiple input arguments: {inputs_inspect}. "
                              f"Please set `call_unpacked=True` to use this model signature for multiple "
                              f"inputs.\nOtherwise, move all inputs into a single `dict` argument and all "
                              f"extra arguments into the `model_kwargs` field.")

            # Can't assume unpacked for single input/output, so warn user if they may be trying to do so
            if len(inputs) == 1 and len(inputs_inspect) == 1 and str(inputs[0]) == str(inputs_inspect[0]):
                warnings.warn(f"Single input argument: {inputs[0]} provided to model with input signature: "
                              f"{inputs_inspect}.\nIf you intended to use a single input argument, set "
                              f"`call_unpacked=True` to use this model signature.\nOtherwise, the first input will "
                              f"be passed to your model as a `dict`.\nIf you are expecting a `dict` input already, "
                              f"change the name of the input to not exactly "
                              f"match {inputs_inspect} in order to silence this warning.")
        # Resolve outputs
        outputs = outputs or []
        outputs = VariableList.merge(outputs, *arg_outputs)
        if len(outputs) == 0:
            outputs = outputs_inspect
            ret_unpacked = True
            if len(outputs) == 0:
                raise ValueError("Could not infer output variables from model inspection. Either your model does not "
                                 "return outputs or an error occurred during inspection.\nPlease provide the "
                                 "outputs directly as `Component(outputs=[...])` or fix the model return values.")
        if ret_unpacked:
            if not all([var == outputs_inspect[i] for i, var in enumerate(outputs)]):
                warnings.warn(f"Mismatch between provided outputs: {outputs.values()} and outputs inferred "
                              f"from model: {outputs_inspect}. This may cause unexpected results.")
        else:
            if len(outputs_inspect) > 1:
                warnings.warn(f"Model expects multiple return values: {outputs_inspect}. Please set "
                              f"`ret_unpacked=True` to use this model signature for multiple outputs.\n"
                              f"Otherwise, move all outputs into a single `dict` return value.")

            if len(outputs) == 1 and len(outputs_inspect) == 1 and str(outputs[0]) == str(outputs_inspect[0]):
                warnings.warn(f"Single output: {outputs[0]} provided to model with single expected return: "
                              f"{outputs_inspect}.\nIf you intended to output a single return value, set "
                              f"`ret_unpacked=True` to use this model signature.\nOtherwise, the output should "
                              f"be returned from your model as a `dict`.\nIf you are returning a `dict` already, "
                              f"then change its name to not exactly match {outputs_inspect} in order to silence "
                              f"this warning.")
        return model, inputs, outputs, call_unpacked, ret_unpacked

    def __repr__(self):
        s = f'---- {self.name} ----\n'
        s += f'Inputs:  {self.inputs}\n'
        s += f'Outputs: {self.outputs}\n'
        s += f'Model:   {self.model}'
        return s

    def __str__(self):
        return self.__repr__()

    @field_validator('serializers')
    @classmethod
    def _validate_serializers(cls, serializers: ComponentSerializers) -> ComponentSerializers:
        for key, serializer in serializers.items():
            if serializer is None:
                serializers[key] = None
            elif isinstance(serializer, YamlSerializable):
                serializers[key] = serializer
            elif isinstance(serializer, str):
                serializers[key] = YamlSerializable.deserialize(serializer)
            else:
                serializers[key] = YamlSerializable(obj=serializer)
        return serializers

    @field_validator('model')
    @classmethod
    def _validate_model(cls, model: str | Callable) -> Callable:
        """Expects model as a callable or a yaml !!python/name string representation."""
        if isinstance(model, str):
            return YamlSerializable.deserialize(model).obj
        else:
            return model

    @field_validator('inputs', 'outputs')
    @classmethod
    def _validate_variables(cls, variables: _VariableLike) -> VariableList:
        if isinstance(variables, VariableList):
            return variables
        else:
            return VariableList.deserialize(variables)

    @field_validator('max_alpha', 'max_beta_train', 'max_beta_interpolator')
    @classmethod
    def _validate_indices(cls, multi_index) -> MultiIndex:
        return MultiIndex(multi_index)

    @field_validator('active_set', 'candidate_set')
    @classmethod
    def _validate_index_set(cls, index_set) -> IndexSet:
        return IndexSet.deserialize(index_set)

    @field_validator('misc_states', 'misc_costs', 'misc_coeff_train', 'misc_coeff_test')
    @classmethod
    def _validate_misc_tree(cls, misc_tree) -> MiscTree:
        return MiscTree.deserialize(misc_tree)

    @field_validator('model_costs')
    @classmethod
    def _validate_model_costs(cls, model_costs: dict) -> dict:
        return {MultiIndex(key): float(value) for key, value in model_costs.items()}

    @field_validator('model_kwargs', 'interpolator', 'training_data')
    @classmethod
    def _validate_arbitrary_serializable(cls, data: Any, info: ValidationInfo) -> Any:
        serializer = info.data.get('serializers').get(info.field_name).obj
        if isinstance(data, Serializable):
            return data
        else:
            return serializer.deserialize(data)

    @field_validator('status')
    @classmethod
    def _validate_status(cls, status: int | SurrogateStatus) -> SurrogateStatus:
        return SurrogateStatus(status)

    @property
    def xdim(self) -> int:
        return len(self.inputs)

    @property
    def ydim(self) -> int:
        return len(self.outputs)

    @property
    def max_beta(self) -> MultiIndex:
        return self.max_beta_train + self.max_beta_interpolator

    @property
    def has_surrogate(self) -> bool:
        """The component has no surrogate model if there are no active or candidate indices."""
        return (len(self.max_alpha) + len(self.max_beta)) > 0

    @property
    def executor(self) -> Executor:
        return self._executor

    @executor.setter
    def executor(self, executor: Executor):
        self._executor = executor

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, logger: logging.Logger):
        self._logger = logger

    def __eq__(self, other):
        if isinstance(other, Component):
            return (self.model.__code__.co_code == other.model.__code__.co_code and self.inputs == other.inputs
                    and self.outputs == other.outputs and self.name == other.name
                    and self.model_kwargs.data == other.model_kwargs.data
                    and self.max_alpha == other.max_alpha and self.max_beta == other.max_beta and
                    self.interpolator == other.interpolator
                    and self.active_set == other.active_set and self.candidate_set == other.candidate_set
                    and self.training_data == other.training_data and self.misc_states == other.misc_states
                    and self.misc_costs == other.misc_costs
                    )
        else:
            return False

    def call_model(self, inputs: dict | Dataset, alpha: Literal['best', 'worst'] | tuple | list = None,
                   output_path: str | Path = None, executor: Executor = None, **kwds) -> Dataset:
        """Wrapper function for calling the underlying component model.

        This function formats the input data, calls the model, and processes the output data.
        It supports vectorized calls, parallel execution using an executor, or serial execution. These options are
        checked in that order, with the first available method used. Must set `Component.vectorized=True` if the
        model supports input arrays of the form `(N,)` or even arbitrary shape `(...,)`.

        !!! Warning "Parallel Execution"
            The underlying model must be defined in a global module scope if `pickle` is the serialization method for
            the provided `Executor`.

        !!! Note "Additional return values"
            The model can return additional items that are not part of `Component.outputs`. These items are returned
            as object arrays in the output `dict`. Two special return values are `model_cost` and `output_path`.
            Returning `model_cost` will store the computational cost of a single model evaluation (which is used by
            `amisc` adaptive surrogate training). Returning `output_path` will store the output file name if the model
            wrote any files to disk.

        !!! Note "Handling errors"
            If the underlying component model raises an exception, the error is stored in `output_dict['errors']` with
            the index of the input data that caused the error. The output data for that index is set to `np.nan`
            for each output variable.

        :param inputs: The input data for the model, formatted as a `dict` with a key for each input variable and
                       a corresponding value that is an array of the input data. If specified as a plain list, then the
                       order is assumed the same as `Component.inputs`.
        :param alpha: Fidelity indices to adjust the model fidelity (model must request this in its keyword arguments).
        :param output_path: Directory to save model output files (model must request this in its keyword arguments).
        :param executor: Executor for parallel execution if the model is not vectorized.
        :param kwds: Additional keyword arguments to pass to the model (model must request these in its keyword args)
        :returns: The output data from the model, formatted as a `dict` with a key for each output variable and a
                  corresponding value that is an array of the output data.
        """
        # Format inputs to a common loop shape (fail if missing any)
        if isinstance(inputs, list | np.ndarray):
            inputs = np.atleast_1d(inputs)
            inputs = {var.name: inputs[..., i] for i, var in enumerate(self.inputs)}
        inputs, loop_shape = format_inputs(inputs, var_shape={var: var.shape for var in self.inputs})
        N = int(np.prod(loop_shape))
        list_alpha = isinstance(alpha, list | np.ndarray)
        alpha_requested = self.model_kwarg_requested('alpha')
        for var in self.inputs:
            if var.name not in inputs:
                raise ValueError(f"Missing input variable '{var.name}'.")

        # Pass extra requested items to the model kwargs
        kwargs = copy.deepcopy(self.model_kwargs.data)
        if self.model_kwarg_requested('output_path'):
            kwargs['output_path'] = output_path
        if self.model_kwarg_requested('input_vars'):
            kwargs['input_vars'] = self.inputs
        if self.model_kwarg_requested('output_vars'):
            kwargs['output_vars'] = self.outputs
        if alpha_requested:
            alpha = np.array(alpha).reshape((N,)) if list_alpha else [alpha] * N
            for i in range(N):
                if alpha[i] == 'best':
                    alpha[i] = tuple(self.max_alpha)
                elif alpha[i] == 'worst':
                    alpha[i] = (0,) * len(self.max_alpha)
        for k, v in kwds.items():
            if self.model_kwarg_requested(k):
                kwargs[k] = v

        # Compute model (vectorized, executor parallel, or serial)
        errors = {}
        if self.vectorized:
            if alpha_requested:
                kwargs['alpha'] = alpha if list_alpha else alpha[0]
            output_dict = self.model(*[inputs[var.name] for var in self.inputs], **kwargs) if self.call_unpacked \
                else self.model(inputs, **kwargs)
            if self.ret_unpacked:
                output_dict = (output_dict,) if not isinstance(output_dict, tuple) else output_dict
                output_dict = {out_var.name: output_dict[i] for i, out_var in enumerate(self.outputs)}
        else:
            executor = executor or self.executor
            if executor is None:  # Serial
                results = deque(maxlen=N)
                for i in range(N):
                    try:
                        if alpha_requested:
                            kwargs['alpha'] = alpha[i]
                        ret = self.model(*[{k: v[i] for k, v in inputs.items()}[var.name] for var in self.inputs],
                                         **kwargs) if self.call_unpacked else (
                            self.model({k: v[i] for k, v in inputs.items()}, **kwargs))
                        if self.ret_unpacked:
                            ret = (ret,) if not isinstance(ret, tuple) else ret
                            ret = {out_var.name: ret[i] for i, out_var in enumerate(self.outputs)}
                        results.append(ret)
                    except Exception as e:
                        results.append({'inputs': {k: v[i] for k, v in inputs.items()}, 'index': i,
                                        'model_kwargs': kwargs.copy(), 'error': str(e)})
            else:  # Parallel
                results = deque(maxlen=N)
                futures = []
                for i in range(N):
                    if alpha_requested:
                        kwargs['alpha'] = alpha[i]
                    fs = executor.submit(self.model,
                                         *[{k: v[i] for k, v in inputs.items()}[var.name] for var in self.inputs],
                                         **kwargs) if self.call_unpacked else (
                        executor.submit(self.model, {k: v[i] for k, v in inputs.items()}, **kwargs))
                    futures.append(fs)
                wait(futures, timeout=None, return_when=ALL_COMPLETED)

                for i, fs in enumerate(futures):
                    try:
                        if alpha_requested:
                            kwargs['alpha'] = alpha[i]
                        ret = fs.result()
                        if self.ret_unpacked:
                            ret = (ret,) if not isinstance(ret, tuple) else ret
                            ret = {out_var.name: ret[i] for i, out_var in enumerate(self.outputs)}
                        results.append(ret)
                    except Exception as e:
                        results.append({'inputs': {k: v[i] for k, v in inputs.items()}, 'index': i,
                                        'model_kwargs': kwargs.copy(), 'error': str(e)})

            # Collect parallel/serial results
            output_dict = {}
            for i in range(N):
                res = results.popleft()
                if 'error' in res:
                    errors[i] = res
                else:
                    for key, val in res.items():
                        if key in self.outputs:
                            if output_dict.get(key) is None:
                                output_dict.setdefault(key, np.full((N, *np.atleast_1d(val).shape), np.nan))
                            output_dict[key][i, ...] = np.atleast_1d(val)
                        elif key == 'model_cost':
                            if output_dict.get(key) is None:
                                output_dict.setdefault(key, np.full((N,), np.nan))
                            output_dict[key][i] = val
                        else:
                            if output_dict.get(key) is None:
                                output_dict.setdefault(key, np.full((N,), None, dtype=object))
                            output_dict[key][i]

        # Save average model costs for each alpha fidelity
        if alpha is not None and output_dict.get('model_cost') is not None:
            alpha_costs = {}
            for i, cost in enumerate(output_dict['model_cost']):
                alpha_costs.setdefault(alpha[i], [])
                alpha_costs[alpha[i]].append(cost)
            for a, costs in alpha_costs.items():
                self.model_costs.setdefault(a, np.empty(0))
                self.model_costs[a] = np.nanmean(np.hstack((costs, self.model_costs[a])))

        # Reshape loop dimensions to match the original input shape
        output_dict = format_outputs(output_dict, loop_shape)

        for var in self.outputs:
            if var.name not in output_dict:
                self.logger.warning(f"Model output missing variable '{var.name}'.")

        # Return the output dictionary and any errors
        if errors:
            output_dict['errors'] = errors
        return output_dict

    def predict(self, x: dict | Dataset, use_model: Literal['best', 'worst'] | tuple = None,
                model_dir: str | Path = None, index_set: Literal['train', 'test'] | IndexSet = 'test',
                misc_coeff: MiscTree = None, incremental: bool = False) -> Dataset:
        """Evaluate the MISC surrogate approximation at new inputs `x`.

        !!! Note
            By default this will predict the MISC surrogate approximation. However, for convenience you can also specify
            `use_model` to call the underlying model function instead. If the component has no surrogate model, then
            the underlying model will be called instead.

        :param x: `dict` of input arrays for each variable input
        :param use_model: 'best'=high-fidelity, 'worst'=low-fidelity, tuple=a specific `alpha`, None=surrogate (default)
        :param model_dir: directory to save output files if `use_model` is specified, ignored otherwise
        :param index_set: the active index set, defaults to `self.active_set` if `'train'` or both
                          `self.active_set + self.candidate_set` if `'test'`
        :param misc_coeff: the data structure holding the MISC coefficients to use, which defaults to the
                           training or testing coefficients depending on the `index_set` parameter.
        :param incremental: a special flag to use if the provided `index_set` is an incremental update to the active
                            index set. A temporary copy of the internal `misc_coeff` data structure will be updated
                            and used to incorporate the new indices.
        :returns: the surrogate approximation of the model (or the model return itself if `use_model`)
        """
        if use_model is not None or not self.has_surrogate:
            return self.call_model(x, alpha=use_model, output_path=model_dir)

        # Choose the correct index set and misc_coeff data structures
        if incremental:
            misc_coeff = copy.deepcopy(self.misc_coeff_train)
            self.update_misc_coeff(index_set, self.active_set, misc_coeff)
            index_set = self.active_set.union(index_set)
        else:
            if isinstance(index_set, str):
                index_set = self.active_set if index_set == 'train' else self.active_set.union(self.candidate_set)
            if misc_coeff is None:
                misc_coeff = self.misc_coeff_train if index_set == 'train' else self.misc_coeff_test

        x, loop_shape = format_inputs(x, var_shape={var: var.shape for var in self.inputs})  # {'x': (N, ...)}
        y = {}

        # Handle prediction with empty active set (return nan)
        if len(index_set) == 0:
            for var in self.outputs:
                y[var] = np.full(loop_shape + var.shape, np.nan)
            return y

        # TODO: Compress input fields and call surrogates, then reconstruct
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[alpha, beta]
            if np.abs(comb_coeff) > 0:
                outputs = self.interpolator.predict(x, self.misc_states.get((alpha, beta)),
                                                    self.training_data.get(alpha, beta[:len(self.max_beta_train)]),
                                                    self.inputs)
                for var, arr in outputs:
                    if y.get(var) is None:
                        y[var] = comb_coeff * arr
                    else:
                        y[var] += comb_coeff * arr

        return format_outputs(y, loop_shape)

    def update_misc_coeff(self, new_indices: IndexSet, index_set: Literal['test', 'train'] | IndexSet = 'train',
                          misc_coeff: MiscTree = None):
        """Update MISC coefficients incrementally resulting from the addition of new indices to an index set.

        !!! Warning "Incremental updates"
            This function is used to update the MISC coefficients stored in `misc_coeff` after adding new indices
            to the given `index_set`. If a custom `index_set` or `misc_coeff` are provided, the user is responsible
            for ensuring the data structures are consistent. Since this is an incremental update, this means all
            existing coefficients for every index in `index_set` should be precomputed and stored in `misc_coeff`.

        :param new_indices: a set of $(\\alpha, \\beta)$ tuples that are being added to the `index_set`
        :param index_set: the active index set, defaults to `self.active_set` if `'train'` or both
                          `self.active_set + self.candidate_set` if `'test'`
        :param misc_coeff: the data structure holding the MISC coefficients to update, which defaults to the
                           training or testing coefficients depending on the `index_set` parameter. This data structure
                           is modified in place.
        """
        if misc_coeff is None:
            match index_set:
                case 'train':
                    misc_coeff = self.misc_coeff_train
                case 'test':
                    misc_coeff = self.misc_coeff_test
                case other:
                    raise ValueError(f"Index set must be 'train' or 'test' if you do not provide `misc_coeff`.")
        if isinstance(index_set, str):
            match index_set:
                case 'train':
                    index_set = self.active_set
                case 'test':
                    index_set = self.active_set.union(self.candidate_set)
                case other:
                    raise ValueError(f"Index set must be 'train' or 'test'.")

        for new_alpha, new_beta in new_indices:
            new_ind = np.array(new_alpha + new_beta)

            # Update all existing/new coefficients if they are a distance of [0, 1] "below" the new index
            # Note that new indices can only be [0, 1] away from themselves -- not any other new indices
            for old_alpha, old_beta in itertools.chain(index_set, [(new_alpha, new_beta)]):
                old_ind = np.array(old_alpha + old_beta)
                diff = new_ind - old_ind
                if np.all(np.isin(diff, [0, 1])):
                    if misc_coeff.get((old_alpha, old_beta)) is None:
                        misc_coeff[old_alpha, old_beta] = 0
                    misc_coeff[old_alpha, old_beta] += (-1) ** int(np.sum(np.abs(diff)))

    def _neighbors(self, alpha: MultiIndex, beta: MultiIndex, active_set: IndexSet = None, forward: bool = True):
        """Get all possible forward or backward multi-index neighbors (distance of one unit vector away)"""
        active_set = active_set or self.active_set
        ind = np.array(alpha + beta)
        max_ind = np.array(self.max_alpha + self.max_beta)
        new_candidates = IndexSet()
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1 if forward else -1

            # Don't add if we surpass a refinement limit or lower bound
            if np.any(ind_new > max_ind) or np.any(ind_new < 0):
                continue

            # Add the new index if it maintains downward-closedness
            down_closed = True
            for j in range(len(ind)):
                ind_check = ind_new.copy()
                ind_check[j] -= 1
                if ind_check[j] >= 0:
                    tup_check = (MultiIndex(ind_check[:len(alpha)]), MultiIndex(ind_check[len(alpha):]))
                    if tup_check not in active_set and tup_check != (alpha, beta):
                        down_closed = False
                        break
            if down_closed:
                new_candidates.add((ind_new[:len(alpha)], ind_new[len(alpha):]))

        return new_candidates

    def refine(self):
        # TODO: wrapper around training data and interpolator refine to manage normalization,
        #  field qty compression, etc.
        # Leave it as a new feature for latent coefficients to have their own normalization
        # Call training_data.initialize if sum(beta) == 0
        pass

    def activate_index(self, alpha: MultiIndex, beta: MultiIndex, model_dir: str | Path = None,
                       executor: Executor = None):
        """Add a multi-index to the active set and all neighbors to the candidate set.

        !!! Warning
            The user of this function is responsible for ensuring that the index set maintains downward-closedness.
            That is, only activate indices that are neighbors of the current active set.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        :param model_dir: Directory to save model output files
        :param executor: Executor for parallel execution of model on training data if the model is not vectorized
        """
        if (alpha, beta) in self.active_set:
            self.logger.warning(f'Multi-index {(alpha, beta)} is already in the active index set. Ignoring...')
            return
        if (alpha, beta) not in self.candidate_set and (sum(alpha) + sum(beta)) > 0:
            # Can only activate the initial index (0, 0, ... 0) without it being in the candidate set
            self.logger.warning(f'Multi-index {(alpha, beta)} is not a neighbor of the active index set, so it '
                                f'cannot be activated. Please only add multi-indices from the candidate set. '
                                f'Ignoring...')
            return

        # Collect all neighbor candidate indices
        executor = executor or self.executor
        neighbors = self._neighbors(alpha, beta, forward=True)
        indices = list(itertools.chain([(alpha, beta)] if (alpha, beta) not in self.candidate_set else [], neighbors))

        # Collect all model inputs (i.e. training points) requested by the new candidates
        alpha_list = []    # keep track of model fidelities
        design_list = []   # keep track of training data coordinates/locations/indices
        model_inputs = {}  # concatenate all model inputs
        field_coords = {}  # keep track of coordinates for field quantities
        for a, b in indices:
            design_idx, design_pts = self.training_data.refine(a, b[:len(self.max_beta_train)], self.inputs)
            alpha_list.extend([tuple(a)] * len(design_idx))
            design_list.append(design_idx)
            for var in self.inputs:
                # Reconstruct latent coefficients and pass full fields (and coords) to the model
                if var.compression is not None:
                    coords = self.model_kwargs.get(f'{var.name}_coords', None)
                    field = var.reconstruct({'latent': design_pts[var]}, coords=coords)
                    coords = field.pop('coords')
                    if field_coords.get(f'{var.name}_coords') is None:
                        field_coords[f'{var.name}_coords'] = copy.deepcopy(coords)

                    for k, v in field.items():
                        model_inputs[k] = v if model_inputs.get(k) is None else (
                            np.concatenate((model_inputs[k], v), axis=0))
                # Normal scalar inputs
                else:
                    model_inputs[var] = design_pts[var] if model_inputs.get(var) is None else (
                        np.concatenate((model_inputs[var], design_pts[var]), axis=0))

        # Evaluate model at designed training points
        self.logger.info(f"Running {len(alpha_list)} total model evaluations for component "
                         f"'{self.name}' new candidate indices: {indices}...")
        model_outputs = self.call_model(model_inputs, alpha=alpha_list, output_path=model_dir, executor=executor,
                                        **field_coords)

        # Unpack model inputs/outputs and update states
        start_idx = 0
        errors = model_outputs.pop('errors', {})
        for i, (a, b) in enumerate(indices):
            num_train_pts = len(design_list[i])
            end_idx = start_idx + num_train_pts
            yi_dict = {var: model_outputs[var][start_idx:end_idx, ...] for var in model_outputs}

            # Check for errors and store
            err_coords = []
            err_list = []
            for idx in list(errors.keys()):
                if idx < end_idx:
                    err_info = errors.pop(idx)
                    err_info['index'] = idx - start_idx
                    err_coords.append(design_list[i][idx - start_idx])
                    err_list.append(err_info)
            if len(err_list) > 0:
                self.logger.warning(f"Model errors occurred while adding candidate ({a}, {b}) for component "
                                    f"{self.name}. Leaving NaN values in training data...")
                self.training_data.set_errors(a, b[:len(self.max_beta_train)], err_coords, err_list)

            # Compress field quantities
            y_vars = []
            for var in self.outputs:
                if var.compression is not None:
                    coords = yi_dict.get(f'{var.name}_coords', None)
                    yi_dict[f'{var.name}_compressed'] = var.compress({field: yi_dict[field] for field in
                                                                      var.compression.fields}, coords=coords)['latent']
                    y_vars.append(f'{var.name}_compressed')
                else:
                    y_vars.append(var.name)

            # Store training data, computational cost, and new interpolator state
            self.training_data.set(a, b[:len(self.max_beta_train)], design_list[i], yi_dict)
            self.training_data.impute_missing_data(a, b[:len(self.max_beta_train)])
            self.misc_costs[a, b] = self.model_costs[a] * num_train_pts
            self.misc_states[a, b] = self.interpolator.refine(b[len(self.max_beta_train):],
                                                              self.training_data.get(a, b[:len(self.max_beta_train)],
                                                                                     y_vars=y_vars),
                                                              self.misc_states.get((alpha, beta)),
                                                              self.inputs)
            start_idx = end_idx

        # Move to the active index set
        s = set()
        s.add((alpha, beta))
        self.update_misc_coeff(IndexSet(s), index_set='train')
        if (alpha, beta) in self.candidate_set:
            self.candidate_set.remove((alpha, beta))
        else:
            # Only for initial index which didn't come from the candidate set
            self.update_misc_coeff(IndexSet(s), index_set='test')
        self.active_set.update(s)

        self.update_misc_coeff(neighbors, index_set='test')  # neighbors will only ever pass through here once
        self.candidate_set.update(neighbors)

    def model_kwarg_requested(self, kwarg_name):
        """Return whether the underlying component model requested this `kwarg_name`. Special kwargs include:

        - `output_path` — a save directory created by `amisc` will be passed to the model for saving model output files.
        - `alpha` — a tuple of model fidelity indices will be passed to the model to adjust fidelity.
        - `input_vars` — a list of `Variable` objects will be passed to the model for input variable information.
        - `output_vars` — a list of `Variable` objects will be passed to the model for output variable information.

        :param kwarg_name: the argument to check for in the underlying component model's function signature kwargs
        """
        signature = inspect.signature(self.model)
        for param in signature.parameters.values():
            if param.name == kwarg_name and param.default != param.empty:
                return True
        return False

    def set_logger(self, log_file: str | Path = None, stdout: bool = None, logger: logging.Logger = None):
        """Set a new `logging.Logger` object.

        :param log_file: log to file (if provided)
        :param stdout: whether to connect the logger to console (defaults to whatever is currently set or False)
        :param logger: the logging object to use (if None, then a new logger is created; this will override
                       the `log_file` and `stdout` arguments if set)
        """
        if stdout is None:
            stdout = False
            if self._logger is not None:
                for handler in self._logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        stdout = True
                        break
        self._logger = logger or get_logger(self.name, log_file=log_file, stdout=stdout)

    def update_model(self, new_model=None, model_kwargs: dict = None, **kwargs):
        """Update the underlying component model or its kwargs."""
        if new_model is not None:
            self.model = new_model
        new_kwargs = self.model_kwargs.data
        new_kwargs.update(model_kwargs or {})
        new_kwargs.update(kwargs)
        self.model_kwargs = new_kwargs

    def get_cost(self, alpha: tuple, beta: tuple) -> float:
        """Return the total cost (wall time s) required to add $(\\alpha, \\beta)$ to the MISC approximation.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        """
        try:
            return self.misc_costs[alpha, beta]
        except Exception:
            return 0.0

    @staticmethod
    def is_downward_closed(indices: IndexSet) -> bool:
        """Return if a list of $(\\alpha, \\beta)$ multi-indices is downward-closed.

        MISC approximations require a downward-closed set in order to use the combination-technique formula for the
        coefficients (as implemented here).

        !!! Example
            The list `[( (0,), (0,) ), ( (1,), (0,) ), ( (1,), (1,) )]` is downward-closed. You can visualize this as
            building a stack of cubes: in order to place a cube, all adjacent cubes must be present (does the logo
            make sense now?).

        :param indices: `list` of (`alpha`, `beta`) multi-indices
        :returns: whether the set of indices is downward-closed
        """
        # Iterate over every multi-index
        for alpha, beta in indices:
            # Every smaller multi-index must also be included in the indices list
            sub_sets = [np.arange(tuple(alpha + beta)[i] + 1) for i in range(len(alpha) + len(beta))]
            for ele in itertools.product(*sub_sets):
                tup = (MultiIndex(ele[:len(alpha)]), MultiIndex(ele[len(alpha):]))
                if tup not in indices:
                    return False
        return True

    def serialize(self, keep_yaml_objects=False, serialize_args=None, serialize_kwargs=None) -> dict:
        """Convert to a `dict` with only standard Python types as fields and values."""
        serialize_args = serialize_args or dict()
        serialize_kwargs = serialize_kwargs or dict()
        d = {}
        for key, value in self.__dict__.items():
            if value is not None and not key.startswith('_'):
                if key == 'serializers':
                    # Update the serializers
                    serializers = self._validate_serializers({k: type(getattr(self, k)) for k in value.keys()})
                    d[key] = {k: (v.obj if keep_yaml_objects else v.serialize()) for k, v in serializers.items()}
                elif key in ['inputs', 'outputs'] and not keep_yaml_objects:
                    d[key] = value.serialize(**serialize_kwargs.get(key, {}))
                elif key == 'model' and not keep_yaml_objects:
                    d[key] = YamlSerializable(obj=value).serialize()
                elif key in ['max_beta_train', 'max_beta_interpolator', 'max_alpha']:
                    if len(value) > 0:
                        d[key] = str(value)
                elif key in ['status']:
                    d[key] = int(value)
                elif key in ['active_set', 'candidate_set']:
                    if len(value) > 0:
                        d[key] = value.serialize()
                elif key in ['misc_costs', 'misc_coeff_train', 'misc_coeff_test', 'misc_states']:
                    if len(value) > 0:
                        d[key] = value.serialize(keep_yaml_objects=keep_yaml_objects)
                elif key in ['model_costs']:
                    if len(value) > 0:
                        d[key] = {str(k): float(v) for k, v in value.items()}
                elif key in ComponentSerializers.__annotations__.keys():
                    d[key] = value.serialize(*serialize_args.get(key, ()), **serialize_kwargs.get(key, {}))
                else:
                    d[key] = value
        return d

    @classmethod
    def deserialize(cls, serialized_data: dict, search_paths=None, search_keys=None) -> Component:
        """Return a `Component` from `data`. Let pydantic handle field validation and conversion. If any component
        data has been saved to file and the save file doesn't exist, then the loader will search for the file
        in the current working directory and any additional search paths provided.

        :param serialized_data: the serialized data to construct the object from
        :param search_paths: paths to try and find any save files (i.e. if they moved since they were serialized),
                             will always search in the current working directory by default
        :param search_keys: keys to search for save files in each component (default is all keys in
                            [`ComponentSerializers`][amisc.component.ComponentSerializers], in addition to variable
                            inputs and outputs)
        """
        if isinstance(serialized_data, Component):
            return serialized_data
        elif callable(serialized_data):
            return cls(serialized_data)  # try to construct a component from a raw model function

        search_paths = search_paths or []
        search_keys = search_keys or []
        search_keys.extend(ComponentSerializers.__annotations__.keys())
        comp = serialized_data

        for key in search_keys:
            if (filename := comp.get(key, None)) is not None:
                comp[key] = search_for_file(filename, search_paths=search_paths)  # will ret original str if not found

        for key in ['inputs', 'outputs']:
            for var in comp.get(key, []):
                if isinstance(var, dict):
                    if (compression := var.get('compression', None)) is not None:
                        var['compression'] = search_for_file(compression, search_paths=search_paths)

        return cls(**comp)

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, comp: Component) -> yaml.MappingNode:
        """Convert a single `Component` object (`data`) to a yaml MappingNode (i.e. a `dict`)."""
        save_path, save_file = _get_yaml_path(dumper)
        serialize_kwargs = {}
        for key, serializer in comp.serializers.items():
            if issubclass(serializer.obj, PickleSerializable):
                filename = save_path / f'{save_file}_{comp.name}_{key}.pkl'
                serialize_kwargs[key] = {'save_path': save_path / filename}
        return dumper.represent_mapping(Component.yaml_tag, comp.serialize(serialize_kwargs=serialize_kwargs,
                                                                           keep_yaml_objects=True))

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the `!Component` tag in yaml to a `Component` object."""
        # Add a file search path in the same directory as the yaml file being loaded from
        save_path, save_file = _get_yaml_path(loader)
        if isinstance(node, yaml.SequenceNode):
            return [ele if isinstance(ele, Component) else Component.deserialize(ele, search_paths=[save_path])
                    for ele in loader.construct_sequence(node, deep=True)]
        elif isinstance(node, yaml.MappingNode):
            return Component.deserialize(loader.construct_mapping(node, deep=True), search_paths=[save_path])
        else:
            raise NotImplementedError(f'The "{Component.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')


class ComponentSurrogate(ABC):
    """The base multi-index stochastic collocation (MISC) surrogate class for a single discipline component model.

    !!! Info "Multi-indices"
        A multi-index is a tuple of natural numbers, each specifying a level of fidelity. You will frequently see two
        multi-indices: `alpha` and `beta`. The `alpha` (or $\\alpha$) indices specify physical model fidelity and get
        passed to the model as an additional argument (e.g. things like discretization level, time step size, etc.).
        The `beta` (or $\\beta$) indices specify surrogate refinement level, so typically an indication of the amount of
        training data used. Each fidelity index in $\\alpha$ and $\\beta$ increase in refinement from $0$ up to
        `max_alpha` and `max_beta`. From the surrogate's perspective, the concatenation of $(\\alpha, \\beta)$ fully
        specifies a single fidelity "level". The `ComponentSurrogate` forms an approximation of the model by summing
        up over many of these concatenated sets of $(\\alpha, \\beta)$. These lists are stored in a data structure of
        `list[ tuple[ tuple, tuple ], ...]`. When $\\alpha$ or $\\beta$ are used as keys in a `dict`, they are cast to
        a Python `str` from a `tuple`.

    :ivar index_set: the current active set of multi-indices in the MISC approximation
    :ivar candidate_set: all neighboring multi-indices that are candidates for inclusion in `index_set`
    :ivar x_vars: list of variables that define the input domain
    :ivar ydim: the number of outputs returned by the model
    :ivar _model: stores a ref to the model or function that is to be approximated, callable as `ret = model(x)`
    :ivar _model_args: additional arguments to supply to the model
    :ivar _model_kwargs: additional keyword arguments to supply to the model
    :ivar truth_alpha: the model fidelity indices to treat as the "ground truth" model
    :ivar max_refine: the maximum level of refinement for each fidelity index in $(\\alpha, \\beta)$
    :ivar surrogates: keeps track of the `BaseInterpolator` associated with each set of $(\\alpha, \\beta)$
    :ivar costs: keeps track of total cost associated with adding a single $(\\alpha, \\beta)$ to the MISC approximation
    :ivar misc_coeff: the combination technique coefficients for the MISC approximation

    :vartype index_set: IndexSet
    :vartype candidate_set: IndexSet
    :vartype x_vars: list[Variable]
    :vartype ydim: int
    :vartype _model: callable[np.ndarray] -> dict
    :vartype _model_args: tuple
    :vartype _model_kwargs: dict
    :vartype truth_alpha: tuple[int, ...]
    :vartype max_refine: list[int, ...]
    :vartype surrogates: MiscTree
    :vartype costs: MiscTree
    :vartype misc_coeff: MiscTree
    """

    def predict(self, x: np.ndarray | float, use_model: str | tuple = None, model_dir: str | Path = None,
                training: bool = False, index_set: IndexSet = None, ppool=None) -> np.ndarray:
        """Evaluate the MISC approximation at new points `x`.

        !!! Note
            By default this will predict the MISC surrogate approximation. However, for convenience you can also specify
            `use_model` to call the underlying function instead.

        :param x: `(..., x_dim)` the points to be interpolated, must be within input domain for accuracy
        :param use_model: 'best'=high-fidelity, 'worst'=low-fidelity, tuple=a specific `alpha`, None=surrogate (default)
        :param model_dir: directory to save output files if `use_model` is specified, ignored otherwise
        :param training: if `True`, then only compute with the active index set, otherwise use all candidates as well
        :param index_set: a list of concatenated $(\\alpha, \\beta)$ to override `self.index_set` if given, else ignore
        :param ppool: a joblib `Parallel` pool to loop over multi-indices in parallel
        :returns y: `(..., y_dim)` the surrogate approximation of the function (or the function itself if `use_model`)
        """
        x = np.atleast_1d(x)
        if use_model is not None:
            return self._bypass_surrogate(x, use_model, model_dir)

        index_set, misc_coeff = self._combination(index_set, training)  # Choose the correct index set and misc_coeff

        def run_batch(alpha, beta, y):
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                func = self.surrogates[str(alpha)][str(beta)]
                y += int(comb_coeff) * func(x)

        if ppool is not None:
            with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd:
                pass
            y_ret = np.memmap(y_fd.name, dtype=x.dtype, mode='r+', shape=x.shape[:-1] + (self.ydim,))
            ppool(delayed(run_batch)(alpha, beta, y_ret) for alpha, beta in index_set)
            y = np.empty(y_ret.shape, dtype=x.dtype)
            y[:] = y_ret[:]
            del y_ret
            os.unlink(y_fd.name)
        else:
            y = np.zeros(x.shape[:-1] + (self.ydim,), dtype=x.dtype)
            for alpha, beta in index_set:
                run_batch(alpha, beta, y)

        return y

    def grad(self, x: np.ndarray | float | list, training: bool = False, index_set: IndexSet = None) -> np.ndarray:
        """Evaluate the derivative/Jacobian of the MISC approximation at new points `x`.

        :param x: `(..., x_dim)` the evaluation points, must be within input domain for accuracy
        :param training: if `True`, then only compute with the active index set, otherwise use all candidates as well
        :param index_set: a list of concatenated $(\\alpha, \\beta)$ to override `self.index_set` if given, else ignore
        :returns: `(..., y_dim, x_dim)` the Jacobian of the surrogate approximation
        """
        x = np.atleast_1d(x)
        index_set, misc_coeff = self._combination(index_set, training)  # Choose the correct index set and misc_coeff

        jac = np.zeros(x.shape[:-1] + (self.ydim, len(self.x_vars)), dtype=x.dtype)
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                interp = self.surrogates[str(alpha)][str(beta)]
                jac += int(comb_coeff) * interp.grad(x)

        return jac

    def hessian(self, x: np.ndarray | float | list, training: bool = False, index_set: IndexSet = None) -> np.ndarray:
        """Evaluate the Hessian of the MISC approximation at new points `x`.

        :param x: `(..., x_dim)` the evaluation points, must be within input domain for accuracy
        :param training: if `True`, then only compute with the active index set, otherwise use all candidates as well
        :param index_set: a list of concatenated $(\\alpha, \\beta)$ to override `self.index_set` if given, else ignore
        :returns: `(..., y_dim, x_dim, x_dim)` the Hessian of the surrogate approximation
        """
        x = np.atleast_1d(x)
        index_set, misc_coeff = self._combination(index_set, training)  # Choose the correct index set and misc_coeff

        hess = np.zeros(x.shape[:-1] + (self.ydim, len(self.x_vars), len(self.x_vars)), x.dtype)
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                interp = self.surrogates[str(alpha)][str(beta)]
                hess += int(comb_coeff) * interp.hessian(x)

        return hess

    def __call__(self, *args, **kwargs):
        """Here for convenience so you can also do `ret = surrogate(x)`, just like the `BaseInterpolator`."""
        return self.predict(*args, **kwargs)


class SparseGridSurrogate(ComponentSurrogate):
    """Concrete MISC surrogate class that maintains a sparse grid composed of smaller tensor-product grids.

    !!! Note
        MISC itself can be thought of as an extension to the well-known sparse grid technique, so this class
        readily integrates with the MISC implementation in `ComponentSurrogate`. Sparse grids limit the curse
        of dimensionality up to about `dim = 10-15` for the input space (which would otherwise be infeasible with a
        normal full tensor-product grid of the same size).

    !!! Info "About points in a sparse grid"
        A sparse grid approximates a full tensor-product grid $(N_1, N_2, ..., N_d)$, where $N_i$ is the number of grid
        points along dimension $i$, for a $d$-dimensional space. Each point is uniquely identified in the sparse grid
        by a list of indices $(j_1, j_2, ..., j_d)$, where $j_i = 0 ... N_i$. We refer to this unique identifier as a
        "grid coordinate". In the `HashSG` data structure, we use a `str(tuple(coord))` representation to uniquely
        identify the coordinate in a hash DS like Python's `dict`.

    :cvar HashSG: a type alias for the hash storage of the sparse grid data (a tree-like DS using dicts)
    :ivar curr_max_beta: the current maximum $\\beta$ refinement indices in the sparse grid (for each $\\alpha$)
    :ivar x_grids: maps $\\alpha$ indices to a list of 1d grids corresponding to `curr_max_beta`
    :ivar xi_map: the sparse grid interpolation points
    :ivar yi_map: the function values at all sparse grid points
    :ivar yi_nan_map: imputed function values to use when `yi_map` contains `nan` data (sometimes the model fails...)
    :ivar yi_files: optional filenames corresponding to the sparse grid `yi_map` data

    :vartype HashSG: dict[str: dict[str: np.ndarray | str]]
    :vartype curr_max_beta: dict[str: list[int]]
    :vartype x_grids: dict[str: np.ndarray]
    :vartype xi_map: HashSG
    :vartype yi_map: HashSG
    :vartype yi_nan_map: HashSG
    :vartype yi_files: HashSG
    """

    HashSG = dict[str: dict[str: np.ndarray | str]]

    def __init__(self, *args, **kwargs):
        # Initialize tree-like hash structures for maintaining a sparse grid of smaller tensor-product grids
        self.curr_max_beta = dict()  # Maps alphas -> current max refinement indices
        self.x_grids = dict()  # Maps alphas -> list of ndarrays specifying 1d grids corresponding to max_beta
        self.xi_map = dict()  # Maps alphas -> grid point coords -> interpolation points
        self.yi_map = dict()  # Maps alphas -> grid point coords -> interpolation function values
        self.yi_nan_map = dict()  # Maps alphas -> grid point coords -> interpolated yi values when yi=nan
        self.yi_files = dict()  # Maps alphas -> grid point coords -> model output files (optional)
        super().__init__(*args, **kwargs)

    # Override
    def predict(self, x, use_model=None, model_dir=None, training=False, index_set=None, ppool=None):
        """Need to override `super()` to allow passing in interpolation grids `xi` and `yi`."""
        x = np.atleast_1d(x)
        if use_model is not None:
            return self._bypass_surrogate(x, use_model, model_dir)

        index_set, misc_coeff = self._combination(index_set, training)

        def run_batch(alpha, beta, y):
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                # Gather the xi/yi interpolation points/qoi_ind for this sub tensor-product grid
                interp = self.surrogates[str(alpha)][str(beta)]
                xi, yi = self.get_tensor_grid(alpha, beta)

                # Add this sub tensor-product grid to the MISC approximation
                y += int(comb_coeff) * interp(x, xi=xi, yi=yi)

        if ppool is not None:
            with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd:
                pass
            y_ret = np.memmap(y_fd.name, dtype=x.dtype, mode='r+', shape=x.shape[:-1] + (self.ydim,))
            ppool(delayed(run_batch)(alpha, beta, y_ret) for alpha, beta in index_set)
            y = np.empty(y_ret.shape, dtype=x.dtype)
            y[:] = y_ret[:]
            del y_ret
            os.unlink(y_fd.name)
        else:
            y = np.zeros(x.shape[:-1] + (self.ydim,), dtype=x.dtype)
            for alpha, beta in index_set:
                run_batch(alpha, beta, y)

        return y

    # Override
    def grad(self, x, training=False, index_set=None):
        """Need to override `super()` to allow passing in interpolation grids `xi` and `yi`."""
        x = np.atleast_1d(x)
        index_set, misc_coeff = self._combination(index_set, training)  # Choose the correct index set and misc_coeff

        jac = np.zeros(x.shape[:-1] + (self.ydim, len(self.x_vars)), dtype=x.dtype)
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                # Gather the xi/yi interpolation points/qoi_ind for this sub tensor-product grid
                interp = self.surrogates[str(alpha)][str(beta)]
                xi, yi = self.get_tensor_grid(alpha, beta)

                jac += int(comb_coeff) * interp.grad(x, xi=xi, yi=yi)

        return jac

    # Override
    def hessian(self, x, training=False, index_set=None):
        """Need to override `super()` to allow passing in interpolation grids `xi` and `yi`."""
        x = np.atleast_1d(x)
        index_set, misc_coeff = self._combination(index_set, training)  # Choose the correct index set and misc_coeff

        hess = np.zeros(x.shape[:-1] + (self.ydim, len(self.x_vars), len(self.x_vars)), dtype=x.dtype)
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                # Gather the xi/yi interpolation points/qoi_ind for this sub tensor-product grid
                interp = self.surrogates[str(alpha)][str(beta)]
                xi, yi = self.get_tensor_grid(alpha, beta)

                hess += int(comb_coeff) * interp.hessian(x, xi=xi, yi=yi)

        return hess

    def get_tensor_grid(self, alpha: tuple, beta: tuple, update_nan: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Construct the `xi/yi` sub tensor-product grids for a given $(\\alpha, \\beta)$ multi-index.

        :param alpha: model fidelity multi-index
        :param beta: surrogate fidelity multi-index
        :param update_nan: try to fill `nan` with imputed values, otherwise just return the `nans` in place
        :returns: `xi, yi`, of size `(prod(grid_sizes), x_dim)` and `(prod(grid_sizes), y_dim)` respectively, the
                  interpolation grid points and corresponding function values for this tensor-product grid
        """
        interp = self.surrogates[str(alpha)][str(beta)]
        grid_sizes = interp.get_grid_sizes(beta)
        coords = [list(range(grid_sizes[n])) for n in range(interp.xdim())]
        xi = np.zeros((np.prod(grid_sizes), interp.xdim()), dtype=np.float32)
        yi = np.zeros((np.prod(grid_sizes), self.ydim), dtype=np.float32)
        for i, coord in enumerate(itertools.product(*coords)):
            xi[i, :] = self.xi_map[str(alpha)][str(coord)]
            yi_curr = self.yi_map[str(alpha)][str(coord)]
            if update_nan and np.any(np.isnan(yi_curr)):
                # Try to replace NaN values if they are stored
                yi_curr = self.yi_nan_map[str(alpha)].get(str(coord), yi_curr)
            yi[i, :] = yi_curr

        return xi, yi

    def get_training_data(self) -> tuple[dict[str: np.ndarray], dict[str: np.ndarray]]:
        """Grab all `x,y` training data stored in the sparse grid for each model fidelity level $\\alpha$.

        :returns: `xi`, `yi`, each a `dict` mapping `alpha` indices to `np.ndarrays`
        """
        xi, yi = dict(), dict()
        for alpha, x_map in self.xi_map.items():
            x = np.zeros((len(x_map), len(self.x_vars)))
            y = np.zeros((len(x_map), self.ydim))
            for i, (coord, x_coord) in enumerate(x_map.items()):
                x[i, :] = x_coord
                y[i, :] = self.yi_nan_map[alpha].get(coord, self.yi_map[alpha][coord])

            xi[alpha] = x
            yi[alpha] = y

        return xi, yi

    def update_yi(self, alpha: tuple, beta: tuple, yi_dict: dict[str: np.ndarray]):
        """Helper method to update `yi` values, accounting for possible `nans` by regression imputation.

        :param alpha: the model fidelity indices
        :param beta: the surrogate fidelity indices
        :param yi_dict: a `dict` mapping `str(coord)` grid coordinates to function values
        """
        self.yi_map[str(alpha)].update(yi_dict)
        imputer, xdim = None, len(self.x_vars)
        for grid_coord, yi in yi_dict.items():
            if np.any(np.isnan(yi)):
                if imputer is None:
                    # Grab all 'good' interpolation points and train a simple linear regression fit
                    xi_mat, yi_mat = np.zeros((0, xdim)), np.zeros((0, self.ydim))
                    for coord, xi in self.xi_map[str(alpha)].items():
                        if coord not in self.yi_nan_map[str(alpha)] and coord in self.yi_map[str(alpha)]:
                            yi_add = self.yi_map[str(alpha)][str(coord)]
                            xi_mat = np.concatenate((xi_mat, xi.reshape((1, xdim))), axis=0)
                            yi_mat = np.concatenate((yi_mat, yi_add.reshape((1, self.ydim))), axis=0)
                    nan_idx = np.any(np.isnan(yi_mat), axis=-1)
                    xi_mat = xi_mat[~nan_idx, :]
                    yi_mat = yi_mat[~nan_idx, :]
                    imputer = Pipeline([('scaler', MaxAbsScaler()), ('model', Ridge(alpha=1))])
                    imputer.fit(xi_mat, yi_mat)
                x_interp = self.xi_map[str(alpha)][str(grid_coord)].reshape((1, xdim))
                y_interp = np.atleast_1d(np.squeeze(imputer.predict(x_interp)))
                nan_idx = np.isnan(yi)
                y_interp[~nan_idx] = yi[~nan_idx]  # Only keep imputed values where yi is nan
                self.yi_nan_map[str(alpha)][str(grid_coord)] = y_interp

        # Go back and try to re-interpolate old nan values as more points get added to the grid
        if imputer is not None:
            for grid_coord in list(self.yi_nan_map[str(alpha)].keys()):
                if grid_coord not in yi_dict:
                    x_interp = self.xi_map[str(alpha)][str(grid_coord)].reshape((1, xdim))
                    y_interp = imputer.predict(x_interp)
                    self.yi_nan_map[str(alpha)][str(grid_coord)] = np.atleast_1d(np.squeeze(y_interp))


class AnalyticalSurrogate(ComponentSurrogate):
    """Concrete "surrogate" class that just uses the analytical model (i.e. bypasses surrogate evaluation)."""

    def __init__(self, x_vars, model, *args, **kwargs):
        """Initializes a stand-in `ComponentSurrogate` with all unnecessary fields set to empty.

        !!! Warning
            This overwrites anything passed in for `truth_alpha`, `max_alpha`, `max_beta`, or `multi_index` since
            these are not used for an analytical model.
        """
        kwargs['truth_alpha'] = ()
        kwargs['max_alpha'] = ()
        kwargs['max_beta'] = ()
        kwargs['multi_index'] = []
        super().__init__(x_vars, model, *args, **kwargs)

    # Override
    def predict(self, x: np.ndarray | float, **kwargs) -> np.ndarray:
        """Evaluate the analytical model at points `x`, ignore extra `**kwargs` passed in.

        :param x: `(..., x_dim)` the points to be evaluated
        :returns y: `(..., y_dim)` the exact model output at the input points
        """
        ret = self._model(x, *self._model_args, **self._model_kwargs)

        if not isinstance(ret, dict):
            self.logger.warning(f"Function {self._model} did not return a dict of the form {{'y': y}}. Please make sure"
                                f" you do so to avoid conflicts. Returning the value directly instead...")

        return ret['y'] if isinstance(ret, dict) else ret

    # Override
    def grad(self, x, training=False, index_set=None):
        """Use auto-diff to compute derivative of an analytical model. Model must be implemented with `numpy`.

        !!! Warning "Not implemented yet"
            Hypothetically, auto-diff libraries like `jax` could be used to write a generic gradient function here for
            analytical models implemented directly in Python/numpy. But there are a lot of quirks that should be worked
            out first.
        """
        raise NotImplementedError('Need to implement a generic auto-diff function here, like using jax for example.')

    def hessian(self, x, training=False, index_set=None):
        """Use auto-diff to compute derivative of an analytical model. Model must be implemented with `numpy`.

        !!! Warning "Not implemented yet"
            Hypothetically, auto-diff libraries like `jax` could be used to write a generic Hessian function here for
            analytical models implemented directly in Python/numpy. But there are a lot of quirks that should be worked
            out first.
        """
        raise NotImplementedError('Need to implement a generic auto-diff function here, like using jax for example.')

    # Override
    def activate_index(self, *args):
        """Do nothing"""
        pass

    # Override
    def add_surrogate(self, *args):
        """Do nothing"""
        pass

    # Override
    def init_coarse(self):
        """Do nothing"""
        pass

    # Override
    def update_misc_coeffs(self, **kwargs):
        """Do nothing"""
        pass

    # Override
    def get_sub_surrogate(self, *args):
        """Nothing to return for analytical model"""
        return None

    # Override
    def get_cost(self, *args):
        """Return no cost for analytical model"""
        return 0

    def build_interpolator(self, *args):
        """Abstract method implementation, return none for an analytical model"""
        return None

    def update_interpolator(self, *args):
        """Abstract method implementation, return `cost=0` for an analytical model"""
        return 0

    def parallel_add_candidates(self, *args):
        """Abstract method implementation, do nothing"""
        pass
