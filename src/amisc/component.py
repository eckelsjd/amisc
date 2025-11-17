"""A `Component` is an `amisc` wrapper around a single discipline model. It manages surrogate construction and
a hierarchy of modeling fidelities.

!!! Info "Multi-indices in the MISC approximation"
    A multi-index is a tuple of natural numbers, each specifying a level of fidelity. You will frequently see two
    multi-indices: `alpha` and `beta`. The `alpha` (or $\\alpha$) indices specify physical model fidelity and get
    passed to the model as an additional argument (e.g. things like discretization level, time step size, etc.).
    The `beta` (or $\\beta$) indices specify surrogate refinement level, so typically an indication of the amount of
    training data used or the complexity of the surrogate model. We divide $\\beta$ into `data_fidelity` and
    `surrogate_fidelity` for specifying training data and surrogate model complexity, respectively.

Includes:

- `ModelKwargs` — a dataclass for storing model keyword arguments
- `StringKwargs` — a dataclass for storing model keyword arguments as a string
- `IndexSet` — a dataclass that maintains a list of multi-indices
- `MiscTree` — a dataclass that maintains MISC data in a `dict` tree, indexed by `alpha` and `beta`
- `Component` — a class that manages a single discipline model and its surrogate hierarchy
"""
from __future__ import annotations

import ast
import copy
import inspect
import itertools
import logging
import random
import string
import time
import traceback
import typing
import warnings
from collections import UserDict, deque
from concurrent.futures import ALL_COMPLETED, Executor, wait
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Literal, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from typing_extensions import TypedDict

from amisc.interpolator import Interpolator, InterpolatorState, Lagrange
from amisc.serialize import PickleSerializable, Serializable, StringSerializable, YamlSerializable
from amisc.training import SparseGrid, TrainingData
from amisc.typing import COORDS_STR_ID, LATENT_STR_ID, Dataset, MultiIndex
from amisc.utils import (
    _get_yaml_path,
    _inspect_assignment,
    _inspect_function,
    format_inputs,
    format_outputs,
    get_logger,
    search_for_file,
    to_model_dataset,
    to_surrogate_dataset,
)
from amisc.variable import Variable, VariableList

__all__ = ["ModelKwargs", "StringKwargs", "IndexSet", "MiscTree", "Component"]
_VariableLike = list[Variable | dict | str] | str | Variable | dict | VariableList  # Generic type for Variables


class ModelKwargs(UserDict, Serializable):
    """Default dataclass for storing model keyword arguments in a `dict`. If you have kwargs that require
    more complicated serialization/specification than a plain `dict`, then you can subclass from here.
    """

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
    elements are formatted correctly as `(alpha, beta)`; that is, as a tuple of `alpha` and
    `beta`, which are themselves instances of a [`MultiIndex`][amisc.typing.MultiIndex] tuple.

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
        """Deserialize a list of tuples to an `IndexSet`."""
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

    The underlying data structure is: `dict[MultiIndex, dict[MultiIndex, float | InterpolatorState]]`.
    """
    SERIALIZER_KEY = 'state_serializer'

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
            ret_dict[self.SERIALIZER_KEY] = state_serializer.obj if keep_yaml_objects else state_serializer.serialize()
        for alpha, beta, data in self:
            ret_dict.setdefault(str(alpha), dict())
            serialized_data = data.serialize(*args, **kwargs) if isinstance(data, InterpolatorState) else float(data)
            ret_dict[str(alpha)][str(beta)] = serialized_data
        return ret_dict

    @classmethod
    def deserialize(cls, serialized_data: dict) -> MiscTree:
        """Deserialize a `dict` to a `MiscTree`.

        :param serialized_data: the data to deserialize to a `MiscTree` object
        """
        return cls(serialized_data)

    @classmethod
    def state_serializer(cls, data: dict) -> YamlSerializable | None:
        """Infer and return the interpolator state serializer from the `MiscTree` data (if possible). If no
        `InterpolatorState` instance could be found, return `None`.
        """
        serializer = data.get(cls.SERIALIZER_KEY, None)  # if `data` is serialized
        if serializer is None:  # Otherwise search for an InterpolatorState
            for alpha, beta_dict in data.items():
                if alpha == cls.SERIALIZER_KEY:
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
            if alpha == cls.SERIALIZER_KEY:
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

    def clear(self):
        """Clear the `MiscTree` data."""
        for key in list(self.data.keys()):
            del self.data[key]

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
            if alpha == self.SERIALIZER_KEY:
                continue
            for beta, data in beta_dict.items():
                yield alpha, beta, data


class ComponentSerializers(TypedDict, total=False):
    """Type hint for the `Component` class data serializers.

    :ivar model_kwargs: the model kwarg object class
    :ivar interpolator: the interpolator object class
    :ivar training_data: the training data object class
    """
    model_kwargs: str | type[Serializable] | YamlSerializable
    interpolator: str | type[Serializable] | YamlSerializable
    training_data: str | type[Serializable] | YamlSerializable


class Component(BaseModel, Serializable):
    """A `Component` wrapper around a single discipline model. It manages MISC surrogate construction and a hierarchy of
    modeling fidelities.

    A `Component` can be constructed by specifying a model, input and output variables, and additional configurations
    such as the maximum fidelity levels, the interpolator type, and the training data type. If `model_fidelity`,
    `data_fidelity`, and `surrogate_fidelity` are all left empty, then the `Component` will not use a surrogate model,
    instead calling the underlying model directly. The `Component` can be serialized to a YAML file and deserialized
    back into a Python object.

    !!! Example "A simple `Component`"
        ```python
        from amisc import Component, Variable

        x = Variable(domain=(0, 1))
        y = Variable()
        model = lambda x: {'y': x['x']**2}
        comp = Component(model=model, inputs=[x], outputs=[y])
        ```

    Each fidelity index in $\\alpha$ increases in refinement from $0$ up to `model_fidelity`. Each fidelity index
    in $\\beta$ increases from $0$ up to `(data_fidelity, surrogate_fidelity)`. From the `Component's` perspective,
    the concatenation of $(\\alpha, \\beta)$ fully specifies a single fidelity "level". The `Component`
    forms an approximation of the model by summing up over many of these concatenated sets of $(\\alpha, \\beta)$.

    :ivar name: the name of the `Component`
    :ivar model: the model or function that is to be approximated, callable as `y = f(x)`
    :ivar inputs: the input variables to the model
    :ivar outputs: the output variables from the model
    :ivar model_kwargs: extra keyword arguments to pass to the model
    :ivar model_fidelity: the maximum level of refinement for each fidelity index in $\\alpha$ for model fidelity
    :ivar data_fidelity: the maximum level of refinement for each fidelity index in $\\beta$ for training data
    :ivar surrogate_fidelity: the max level of refinement for each fidelity index in $\\beta$ for the surrogate
    :ivar interpolator: the interpolator to use as the underlying surrogate model
    :ivar vectorized: whether the model supports vectorized input/output (i.e. datasets with arbitrary shape `(...,)`)
    :ivar call_unpacked: whether the model expects unpacked input arguments (i.e. `func(x1, x2, ...)`)
    :ivar ret_unpacked: whether the model returns unpacked output arguments (i.e. `func() -> (y1, y2, ...)`)

    :ivar active_set: the current active set of multi-indices in the MISC approximation
    :ivar candidate_set: all neighboring multi-indices that are candidates for inclusion in `active_set`
    :ivar misc_states: the interpolator states for each multi-index in the MISC approximation
    :ivar misc_costs: the computational cost associated with each multi-index in the MISC approximation
    :ivar misc_coeff_train: the combination technique coefficients for the active set multi-indices
    :ivar misc_coeff_test: the combination technique coefficients for the active and candidate set multi-indices
    :ivar model_costs: the tracked average single fidelity model costs for each $\\alpha$
    :ivar model_evals: the tracked number of evaluations for each $\\alpha$
    :ivar training_data: the training data storage structure for the surrogate model

    :ivar serializers: the custom serializers for the `[model_kwargs, interpolator, training_data]`
                       `Component` attributes -- these should be the _types_ of the serializer objects, which will
                       be inferred from the data passed in if not explicitly set
    :ivar _logger: the logger for the `Component`
    """
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
    model_fidelity: str | tuple = MultiIndex()
    data_fidelity: str | tuple = MultiIndex()
    surrogate_fidelity: str | tuple = MultiIndex()
    interpolator: Any | Interpolator = Lagrange()
    vectorized: bool = False
    call_unpacked: Optional[bool] = None  # If the model expects inputs/outputs like `func(x1, x2, ...)->(y1, y2, ...)
    ret_unpacked: Optional[bool] = None

    # Data storage/states for a MISC component
    active_set: list | set | IndexSet = IndexSet()     # set of active (alpha, beta) multi-indices
    candidate_set: list | set | IndexSet = IndexSet()  # set of candidate (alpha, beta) multi-indices
    misc_states: dict | MiscTree = MiscTree()          # (alpha, beta) -> Interpolator state
    misc_costs: dict | MiscTree = MiscTree()           # (alpha, beta) -> Added computational cost for this mult-index
    misc_coeff_train: dict | MiscTree = MiscTree()     # (alpha, beta) -> c_[alpha, beta] (active set only)
    misc_coeff_test: dict | MiscTree = MiscTree()      # (alpha, beta) -> c_[alpha, beta] (including candidate set)
    model_costs: dict = dict()                         # Average single fidelity model costs (for each alpha)
    model_evals: dict = dict()                         # Number of evaluations for each alpha
    training_data: Any | TrainingData = SparseGrid()   # Stores surrogate training data

    # Internal
    _logger: Optional[logging.Logger] = None
    _model_start_time: float = -1.0  # Temporarily store the most recent model start timestamp from call_model
    _model_end_time: float = -1.0    # Temporarily store the most recent model end timestamp from call_model
    _cache: dict = dict()            # Temporary cache for faster access to training data and similar

    def __init__(self, /, model, *args, inputs=None, outputs=None, name=None, **kwargs):
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
                kwargs[key] = field
            if not serializers.get(key, None):
                serializers[key] = type(field) if isinstance(field, Serializable) else (
                    type(self.model_fields[key].default))
        kwargs['serializers'] = serializers

        super().__init__(model=model, inputs=inputs, outputs=outputs, name=name, **kwargs)  # Runs pydantic validation

        # Set internal properties
        assert self.is_downward_closed(self.active_set.union(self.candidate_set))
        self.set_logger()

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
        """Make sure custom serializer object types are themselves serializable as `YamlSerializable`."""
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

    @field_validator('model_fidelity', 'data_fidelity', 'surrogate_fidelity')
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

    @field_validator('model_evals')
    @classmethod
    def _validate_model_evals(cls, model_evals: dict) -> dict:
        return {MultiIndex(key): int(value) for key, value in model_evals.items()}

    @field_validator('model_kwargs', 'interpolator', 'training_data')
    @classmethod
    def _validate_arbitrary_serializable(cls, data: Any, info: ValidationInfo) -> Any:
        """Use the stored custom serialization classes to deserialize arbitrary objects."""
        serializer = info.data.get('serializers').get(info.field_name).obj
        if isinstance(data, Serializable):
            return data
        else:
            return serializer.deserialize(data)

    @property
    def xdim(self) -> int:
        return len(self.inputs)

    @property
    def ydim(self) -> int:
        return len(self.outputs)

    @property
    def max_alpha(self) -> MultiIndex:
        """The maximum model fidelity multi-index (alias for `model_fidelity`)."""
        return self.model_fidelity

    @property
    def max_beta(self) -> MultiIndex:
        """The maximum surrogate fidelity multi-index is a combination of training and interpolator indices."""
        return self.data_fidelity + self.surrogate_fidelity

    @property
    def has_surrogate(self) -> bool:
        """The component has no surrogate model if there are no fidelity indices."""
        return (len(self.max_alpha) + len(self.max_beta)) > 0

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
                    and self.model_fidelity == other.model_fidelity and self.max_beta == other.max_beta and
                    self.interpolator == other.interpolator
                    and self.active_set == other.active_set and self.candidate_set == other.candidate_set
                    and self.misc_states == other.misc_states and self.misc_costs == other.misc_costs
                    )
        else:
            return False

    def _neighbors(self, alpha: MultiIndex, beta: MultiIndex, active_set: IndexSet = None, forward: bool = True):
        """Get all possible forward or backward multi-index neighbors (distance of one unit vector away).

        :param alpha: the model fidelity index
        :param beta: the surrogate fidelity index
        :param active_set: the set of active multi-indices
        :param forward: whether to get forward or backward neighbors
        :returns: a set of multi-indices that are neighbors of the input multi-index pair `(alpha, beta)`
        """
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

    def _surrogate_outputs(self):
        """Helper function to get the names of the surrogate outputs (including latent variables)."""
        y_vars = []
        for var in self.outputs:
            if var.compression is not None:
                for i in range(var.compression.latent_size()):
                    y_vars.append(f'{var.name}{LATENT_STR_ID}{i}')
            else:
                y_vars.append(var.name)
        return y_vars

    def _match_index_set(self, index_set, misc_coeff):
        """Helper function to grab the correct data structures for the given index set and MISC coefficients."""
        if misc_coeff is None:
            match index_set:
                case 'train':
                    misc_coeff = self.misc_coeff_train
                case 'test':
                    misc_coeff = self.misc_coeff_test
                case other:
                    raise ValueError(f"Index set must be 'train' or 'test' if you do not provide `misc_coeff`. "
                                     f"{other} not recognized.")
        if isinstance(index_set, str):
            match index_set:
                case 'train':
                    index_set = self.active_set
                case 'test':
                    index_set = self.active_set.union(self.candidate_set)
                case other:
                    raise ValueError(f"Index set must be 'train' or 'test'. {other} not recognized.")

        return index_set, misc_coeff

    def cache(self, kind: list | Literal["training"] = "training"):
        """Cache data for quicker access. Only `"training"` is supported.

        :param kind: the type(s) of data to cache (only "training" is supported). This will cache the
                     surrogate training data with nans removed.
        """
        if not isinstance(kind, list):
            kind = [kind]

        if "training" in kind:
            self._cache.setdefault("training", {})
            y_vars = self._surrogate_outputs()
            for alpha, beta in self.active_set.union(self.candidate_set):
                self._cache["training"].setdefault(alpha, {})

                if beta not in self._cache["training"][alpha]:
                    self._cache["training"][alpha][beta] = self.training_data.get(alpha, beta[:len(self.data_fidelity)],
                                                                                  y_vars=y_vars, skip_nan=True)

    def clear_cache(self):
        """Clear cached data."""
        self._cache.clear()

    def get_training_data(self, alpha: Literal['best', 'worst'] | MultiIndex = 'best',
                          beta: Literal['best', 'worst'] | MultiIndex = 'best',
                          y_vars: list = None,
                          cached: bool = False) -> tuple[Dataset, Dataset]:
        """Get all training data for a given multi-index pair `(alpha, beta)`.

        :param alpha: the model fidelity index (defaults to the maximum available model fidelity)
        :param beta: the surrogate fidelity index (defaults to the maximum available surrogate fidelity)
        :param y_vars: the training data to return (defaults to all stored data)
        :param cached: if True, will get cached training data if available (this will ignore `y_vars` and
                       only grab whatever is in the cache, which is surrogate outputs only and no nans)
        :returns: `(xtrain, ytrain)` - the training data for the given multi-indices
        """
        # Find the best alpha
        if alpha == 'best':
            alpha_best = ()
            for a, _ in self.active_set.union(self.candidate_set):
                if sum(a) > sum(alpha_best):
                    alpha_best = a
            alpha = alpha_best
        elif alpha == 'worst':
            alpha = (0,) * len(self.max_alpha)

        # Find the best beta for the given alpha
        if beta == 'best':
            beta_best = ()
            for a, b in self.active_set.union(self.candidate_set):
                if a == alpha and sum(b) > sum(beta_best):
                    beta_best = b
            beta = beta_best
        elif beta == 'worst':
            beta = (0,) * len(self.max_beta)

        try:
            if cached and (data := self._cache.get("training", {}).get(alpha, {}).get(beta)) is not None:
                return data
            else:
                return self.training_data.get(alpha, beta[:len(self.data_fidelity)], y_vars=y_vars, skip_nan=True)
        except Exception as e:
            self.logger.error(f"Error getting training data for alpha={alpha}, beta={beta}.")
            raise e

    def call_model(self, inputs: dict | Dataset,
                   model_fidelity: Literal['best', 'worst'] | tuple | list = None,
                   output_path: str | Path = None,
                   executor: Executor = None,
                   track_costs: bool = False,
                   **kwds) -> Dataset:
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
        :param model_fidelity: Fidelity indices to tune the model fidelity (model must request this
                               in its keyword arguments).
        :param output_path: Directory to save model output files (model must request this in its keyword arguments).
        :param executor: Executor for parallel execution if the model is not vectorized (optional).
        :param track_costs: Whether to track the computational cost of each model evaluation.
        :param kwds: Additional keyword arguments to pass to the model (model must request these in its keyword args).
        :returns: The output data from the model, formatted as a `dict` with a key for each output variable and a
                  corresponding value that is an array of the output data.
        """
        # Format inputs to a common loop shape (fail if missing any)
        if len(inputs) == 0:
            return {}  # your fault
        if isinstance(inputs, list | np.ndarray):
            inputs = np.atleast_1d(inputs)
            inputs = {var.name: inputs[..., i] for i, var in enumerate(self.inputs)}

        var_shape = {}
        for var in self.inputs:
            s = None
            if (arr := kwds.get(f'{var.name}{COORDS_STR_ID}')) is not None:
                if not np.issubdtype(arr.dtype, np.object_):  # if not object array, then it's a single coordinate set
                    s = arr.shape if len(arr.shape) == 1 else arr.shape[:-1]  # skip the coordinate dim (last axis)
            if var.compression is not None:
                for field in var.compression.fields:
                    var_shape[field] = s
            else:
                var_shape[var.name] = s
        inputs, loop_shape = format_inputs(inputs, var_shape=var_shape)

        N = int(np.prod(loop_shape))
        list_alpha = isinstance(model_fidelity, list | np.ndarray)
        alpha_requested = self.model_kwarg_requested('model_fidelity')
        for var in self.inputs:
            if var.compression is not None:
                for field in var.compression.fields:
                    if field not in inputs:
                        raise ValueError(f"Missing field '{field}' for input variable '{var}'.")
            elif var.name not in inputs:
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
            if not list_alpha:
                model_fidelity = [model_fidelity] * N
            for i in range(N):
                if model_fidelity[i] == 'best':
                    model_fidelity[i] = self.max_alpha
                elif model_fidelity[i] == 'worst':
                    model_fidelity[i] = (0,) * len(self.model_fidelity)

        for k, v in kwds.items():
            if self.model_kwarg_requested(k):
                kwargs[k] = v

        # Compute model (vectorized, executor parallel, or serial)
        errors = {}
        if self.vectorized:
            if alpha_requested:
                kwargs['model_fidelity'] = np.atleast_1d(model_fidelity).reshape((N, -1))

            self._model_start_time = time.time()
            output_dict = self.model(*[inputs[var.name] for var in self.inputs], **kwargs) if self.call_unpacked \
                else self.model(inputs, **kwargs)
            self._model_end_time = time.time()

            if self.ret_unpacked:
                output_dict = (output_dict,) if not isinstance(output_dict, tuple) else output_dict
                output_dict = {out_var.name: output_dict[i] for i, out_var in enumerate(self.outputs)}
        else:
            self._model_start_time = time.time()
            if executor is None:  # Serial
                results = deque(maxlen=N)
                for i in range(N):
                    try:
                        if alpha_requested:
                            kwargs['model_fidelity'] = model_fidelity[i]
                        ret = self.model(*[{k: v[i] for k, v in inputs.items()}[var.name] for var in self.inputs],
                                         **kwargs) if self.call_unpacked else (
                            self.model({k: v[i] for k, v in inputs.items()}, **kwargs))
                        if self.ret_unpacked:
                            ret = (ret,) if not isinstance(ret, tuple) else ret
                            ret = {out_var.name: ret[i] for i, out_var in enumerate(self.outputs)}
                        results.append(ret)
                    except Exception:
                        results.append({'inputs': {k: v[i] for k, v in inputs.items()}, 'index': i,
                                        'model_kwargs': kwargs.copy(), 'error': traceback.format_exc()})
            else:  # Parallel
                results = deque(maxlen=N)
                futures = []
                for i in range(N):
                    if alpha_requested:
                        kwargs['model_fidelity'] = model_fidelity[i]
                    fs = executor.submit(self.model,
                                         *[{k: v[i] for k, v in inputs.items()}[var.name] for var in self.inputs],
                                         **kwargs) if self.call_unpacked else (
                        executor.submit(self.model, {k: v[i] for k, v in inputs.items()}, **kwargs))
                    futures.append(fs)
                wait(futures, timeout=None, return_when=ALL_COMPLETED)

                for i, fs in enumerate(futures):
                    try:
                        if alpha_requested:
                            kwargs['model_fidelity'] = model_fidelity[i]
                        ret = fs.result()
                        if self.ret_unpacked:
                            ret = (ret,) if not isinstance(ret, tuple) else ret
                            ret = {out_var.name: ret[i] for i, out_var in enumerate(self.outputs)}
                        results.append(ret)
                    except Exception:
                        results.append({'inputs': {k: v[i] for k, v in inputs.items()}, 'index': i,
                                        'model_kwargs': kwargs.copy(), 'error': traceback.format_exc()})
            self._model_end_time = time.time()

            # Collect parallel/serial results
            output_dict = {}
            for i in range(N):
                res = results.popleft()
                if 'error' in res:
                    errors[i] = res
                else:
                    for key, val in res.items():
                        # Save this component's variables
                        is_component_var = False
                        for var in self.outputs:
                            if var.compression is not None:  # field quantity return values (save as object arrays)
                                if key in var.compression.fields or key == f'{var}{COORDS_STR_ID}':
                                    if output_dict.get(key) is None:
                                        output_dict.setdefault(key, np.full((N,), None, dtype=object))
                                    output_dict[key][i] = np.atleast_1d(val)
                                    is_component_var = True
                                    break
                            elif key == var:
                                if output_dict.get(key) is None:
                                    _val = np.atleast_1d(val)
                                    _extra_shape = () if len(_val.shape) == 1 and _val.shape[0] == 1 else _val.shape
                                    output_dict.setdefault(key, np.full((N, *_extra_shape), np.nan))
                                output_dict[key][i, ...] = np.atleast_1d(val)
                                is_component_var = True
                                break

                        # Otherwise, save other objects
                        if not is_component_var:
                            # Save singleton numeric values as numeric arrays (model costs, etc.)
                            _val = np.atleast_1d(val)
                            if key == 'model_cost' or (np.issubdtype(_val.dtype, np.number)
                                                       and len(_val.shape) == 1 and _val.shape[0] == 1):
                                if output_dict.get(key) is None:
                                    output_dict.setdefault(key, np.full((N,), np.nan))
                                output_dict[key][i] = _val[0]
                            else:
                                # Otherwise save into a generic object array
                                if output_dict.get(key) is None:
                                    output_dict.setdefault(key, np.full((N,), None, dtype=object))
                                output_dict[key][i] = val

        # Save average model costs for each alpha fidelity
        if track_costs:
            if model_fidelity is not None and output_dict.get('model_cost') is not None:
                alpha_costs = {}
                for i, cost in enumerate(output_dict['model_cost']):
                    alpha_costs.setdefault(MultiIndex(model_fidelity[i]), [])
                    alpha_costs[MultiIndex(model_fidelity[i])].append(cost)
                for a, costs in alpha_costs.items():
                    self.model_evals.setdefault(a, 0)
                    self.model_costs.setdefault(a, 0.0)
                    num_evals_prev = self.model_evals.get(a)
                    num_evals_new = len(costs)
                    prev_avg = self.model_costs.get(a)
                    costs = np.nan_to_num(costs, nan=prev_avg)
                    new_avg = (np.sum(costs) + prev_avg * num_evals_prev) / (num_evals_prev + num_evals_new)
                    self.model_evals[a] += num_evals_new
                    self.model_costs[a] = float(new_avg)

        # Reshape loop dimensions to match the original input shape
        output_dict = format_outputs(output_dict, loop_shape)

        for var in self.outputs:
            if var.compression is not None:
                for field in var.compression.fields:
                    if field not in output_dict:
                        self.logger.warning(f"Model return missing field '{field}' for output variable '{var}'. "
                                            f"This may indicate an error during model evaluation. Returning NaNs...")
                        output_dict.setdefault(field, np.full((N,), np.nan))
            elif var.name not in output_dict:
                self.logger.warning(f"Model return missing output variable '{var.name}'. This may indicate "
                                    f"an error during model evaluation. Returning NaNs...")
                output_dict[var.name] = np.full((N,), np.nan)

        # Return the output dictionary and any errors
        if errors:
            output_dict['errors'] = errors
        return output_dict

    def predict(self, inputs: dict | Dataset,
                use_model: Literal['best', 'worst'] | tuple = None,
                model_dir: str | Path = None,
                index_set: Literal['train', 'test'] | IndexSet = 'test',
                misc_coeff: MiscTree = None,
                incremental: bool = False,
                executor: Executor = None,
                **kwds) -> Dataset:
        """Evaluate the MISC surrogate approximation at new inputs `x`.

        !!! Note "Using the underlying model"
            By default this will predict the MISC surrogate approximation; all inputs are assumed to be in a compressed
            and normalized form. If the component does not have a surrogate (i.e. it is analytical), then the inputs
            will be converted to model form and the underlying model will be called in place. If you instead want to
            override the surrogate, passing `use_model` will call the underlying model directly. In that case, the
            inputs should be passed in already in model form (i.e. full fields, denormalized).

        :param inputs: `dict` of input arrays for each variable input
        :param use_model: 'best'=high-fidelity, 'worst'=low-fidelity, tuple=a specific `alpha`, None=surrogate (default)
        :param model_dir: directory to save output files if `use_model` is specified, ignored otherwise
        :param index_set: the active index set, defaults to `self.active_set` if `'train'` or both
                          `self.active_set + self.candidate_set` if `'test'`
        :param misc_coeff: the data structure holding the MISC coefficients to use, which defaults to the
                           training or testing coefficients depending on the `index_set` parameter.
        :param incremental: a special flag to use if the provided `index_set` is an incremental update to the active
                            index set. A temporary copy of the internal `misc_coeff` data structure will be updated
                            and used to incorporate the new indices.
        :param executor: executor for parallel execution if the model is not vectorized (optional), will use the
                         executor for looping over MISC coefficients if evaluating the surrogate rather than the model
        :param kwds: additional keyword arguments to pass to the model (if using the underlying model)
        :returns: the surrogate approximation of the model (or the model return itself if `use_model`)
        """
        # Use raw model inputs/outputs
        if use_model is not None:
            outputs = self.call_model(inputs, model_fidelity=use_model, output_path=model_dir, executor=executor,**kwds)
            return {str(var): outputs[var] for var in outputs}

        # Convert inputs/outputs to/from model if no surrogate (i.e. analytical models)
        if not self.has_surrogate:
            field_coords = {f'{var}{COORDS_STR_ID}':
                            self.model_kwargs.get(f'{var}{COORDS_STR_ID}', kwds.get(f'{var}{COORDS_STR_ID}', None))
                            for var in self.inputs}
            inputs, field_coords = to_model_dataset(inputs, self.inputs, del_latent=True, **field_coords)
            field_coords.update(kwds)
            outputs = self.call_model(inputs, model_fidelity=use_model or 'best', output_path=model_dir,
                                      executor=executor, **field_coords)
            outputs, _ = to_surrogate_dataset(outputs, self.outputs, del_fields=True, **field_coords)
            return {str(var): outputs[var] for var in outputs}

        # Choose the correct index set and misc_coeff data structures
        if incremental:
            misc_coeff = copy.deepcopy(self.misc_coeff_train)
            self.update_misc_coeff(index_set, self.active_set, misc_coeff)
            index_set = self.active_set.union(index_set)
        else:
            index_set, misc_coeff = self._match_index_set(index_set, misc_coeff)

        # Format inputs for surrogate prediction (all scalars at this point, including latent coeffs)
        inputs, loop_shape = format_inputs(inputs)  # {'x': (N,)}
        outputs = {}

        # Handle prediction with empty active set (return nan)
        if len(index_set) == 0:
            self.logger.warning(f"Component '{self.name}' has an empty active set. "
                                f"Has the surrogate been trained yet? Returning NaNs...")
            for var in self.outputs:
                outputs[var.name] = np.full(loop_shape, np.nan)
            return outputs

        y_vars = self._surrogate_outputs()  # Only request this component's specified outputs (ignore all extras)

        # Combination technique MISC surrogate prediction
        results = []
        coeffs = []
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[alpha, beta]
            if np.abs(comb_coeff) > 0:
                coeffs.append(comb_coeff)
                args = (self.misc_states.get((alpha, beta)),
                        self.get_training_data(alpha, beta, y_vars=y_vars, cached=True))

                results.append(self.interpolator.predict(inputs, *args) if executor is None else
                               executor.submit(self.interpolator.predict, inputs, *args))

        if executor is not None:
            wait(results, timeout=None, return_when=ALL_COMPLETED)
            results = [future.result() for future in results]

        for coeff, interp_pred in zip(coeffs, results):
            for var, arr in interp_pred.items():
                if outputs.get(var) is None:
                    outputs[str(var)] = coeff * arr
                else:
                    outputs[str(var)] += coeff * arr

        return format_outputs(outputs, loop_shape)

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
        index_set, misc_coeff = self._match_index_set(index_set, misc_coeff)

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

    def activate_index(self, alpha: MultiIndex, beta: MultiIndex, model_dir: str | Path = None,
                       executor: Executor = None, weight_fcns: dict[str, callable] | Literal['pdf'] | None = 'pdf'):
        """Add a multi-index to the active set and all neighbors to the candidate set.

        !!! Warning
            The user of this function is responsible for ensuring that the index set maintains downward-closedness.
            That is, only activate indices that are neighbors of the current active set.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        :param model_dir: Directory to save model output files
        :param executor: Executor for parallel execution of model on training data if the model is not vectorized
        :param weight_fcns: Dictionary of weight functions for each input variable (defaults to the variable PDFs);
                            each function should be callable as `fcn(x: np.ndarray) -> np.ndarray`, where the input
                            is an array of normalized input data and the output is an array of weights. If None, then
                            no weighting is applied.
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

        # Collect all neighbor candidate indices; sort by largest model cost first
        neighbors = self._neighbors(alpha, beta, forward=True)
        indices = list(itertools.chain([(alpha, beta)] if (alpha, beta) not in self.candidate_set else [], neighbors))
        indices.sort(key=lambda ele: self.model_costs.get(ele[0], sum(ele[0])), reverse=True)

        # Refine and collect all new model inputs (i.e. training points) requested by the new candidates
        alpha_list = []    # keep track of model fidelities
        design_list = []   # keep track of training data coordinates/locations/indices
        model_inputs = {}  # concatenate all model inputs
        field_coords = {f'{var}{COORDS_STR_ID}': self.model_kwargs.get(f'{var}{COORDS_STR_ID}', None)
                        for var in self.inputs}
        domains = self.inputs.get_domains()

        if weight_fcns == 'pdf':
            weight_fcns = self.inputs.get_pdfs()

        for a, b in indices:
            if ((a, b[:len(self.data_fidelity)] + (0,) * len(self.surrogate_fidelity)) in
                    self.active_set.union(self.candidate_set)):
                # Don't refine training data if only updating surrogate fidelity indices
                # Training data is the same for all surrogate fidelity indices, given constant data fidelity
                design_list.append([])
                continue

            design_coords, design_pts = self.training_data.refine(a, b[:len(self.data_fidelity)],
                                                                  domains, weight_fcns)
            design_pts, fc = to_model_dataset(design_pts, self.inputs, del_latent=True, **field_coords)

            # Remove duplicate (alpha, coords) pairs -- so you don't evaluate the model twice for the same input
            i = 0
            del_idx = []
            for other_design in design_list:
                for other_coord in other_design:
                    for j, curr_coord in enumerate(design_coords):
                        if curr_coord == other_coord and a == alpha_list[i] and j not in del_idx:
                            del_idx.append(j)
                    i += 1
            design_coords = [design_coords[j] for j in range(len(design_coords)) if j not in del_idx]
            design_pts = {var: np.delete(arr, del_idx, axis=0) for var, arr in design_pts.items()}

            alpha_list.extend([tuple(a)] * len(design_coords))
            design_list.append(design_coords)
            field_coords.update(fc)
            for var in design_pts:
                model_inputs[var] = design_pts[var] if model_inputs.get(var) is None else (
                    np.concatenate((model_inputs[var], design_pts[var]), axis=0))

        # Evaluate model at designed training points
        if len(alpha_list) > 0:
            self.logger.info(f"Running {len(alpha_list)} total model evaluations for component "
                             f"'{self.name}' new candidate indices: {indices}...")
            model_outputs = self.call_model(model_inputs, model_fidelity=alpha_list, output_path=model_dir,
                                            executor=executor, track_costs=True, **field_coords)
            self.logger.info(f"Model evaluations complete for component '{self.name}'.")
            errors = model_outputs.pop('errors', {})
        else:
            self._model_start_time = -1.0
            self._model_end_time = -1.0

        # Unpack model outputs and update states
        start_idx = 0
        for i, (a, b) in enumerate(indices):
            num_train_pts = len(design_list[i])
            end_idx = start_idx + num_train_pts  # Ensure loop dim of 1 gets its own axis (might have been squeezed)

            if num_train_pts > 0:
                yi_dict = {var: arr[np.newaxis, ...] if len(alpha_list) == 1 and arr.shape[0] != 1 else
                           arr[start_idx:end_idx, ...] for var, arr in model_outputs.items()}

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
                    self.training_data.set_errors(a, b[:len(self.data_fidelity)], err_coords, err_list)

                # Compress field quantities and normalize
                yi_dict, y_vars = to_surrogate_dataset(yi_dict, self.outputs, del_fields=False, **field_coords)

                # Store training data, computational cost, and new interpolator state
                self.training_data.set(a, b[:len(self.data_fidelity)], design_list[i], yi_dict)
                self.training_data.impute_missing_data(a, b[:len(self.data_fidelity)])

            else:
                y_vars = self._surrogate_outputs()

            self.misc_costs[a, b] = num_train_pts
            self.misc_states[a, b] = self.interpolator.refine(b[len(self.data_fidelity):],
                                                              self.training_data.get(a, b[:len(self.data_fidelity)],
                                                                                     y_vars=y_vars, skip_nan=True),
                                                              self.misc_states.get((alpha, beta)),
                                                              domains)
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

    def gradient(self, inputs: dict | Dataset,
                 index_set: Literal['train', 'test'] | IndexSet = 'test',
                 misc_coeff: MiscTree = None,
                 derivative: Literal['first', 'second'] = 'first',
                 executor: Executor = None) -> Dataset:
        """Evaluate the Jacobian or Hessian of the MISC surrogate approximation at new `inputs`, i.e.
        the first or second derivatives, respectively.

        :param inputs: `dict` of input arrays for each variable input
        :param index_set: the active index set, defaults to `self.active_set` if `'train'` or both
                          `self.active_set + self.candidate_set` if `'test'`
        :param misc_coeff: the data structure holding the MISC coefficients to use, which defaults to the
                           training or testing coefficients depending on the `index_set` parameter.
        :param derivative: whether to compute the first or second derivative (i.e. Jacobian or Hessian)
        :param executor: executor for looping over MISC coefficients (optional)
        :returns: a `dict` of the Jacobian or Hessian of the surrogate approximation for each output variable
        """
        if not self.has_surrogate:
            self.logger.warning("No surrogate model available for gradient computation.")
            return None

        index_set, misc_coeff = self._match_index_set(index_set, misc_coeff)
        inputs, loop_shape = format_inputs(inputs)  # {'x': (N,)}
        outputs = {}

        if len(index_set) == 0:
            for var in self.outputs:
                outputs[var] = np.full(loop_shape, np.nan)
            return outputs
        y_vars = self._surrogate_outputs()

        # Combination technique MISC gradient prediction
        results = []
        coeffs = []
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[alpha, beta]
            if np.abs(comb_coeff) > 0:
                coeffs.append(comb_coeff)
                func = self.interpolator.gradient if derivative == 'first' else self.interpolator.hessian
                args = (self.misc_states.get((alpha, beta)),
                        self.get_training_data(alpha, beta, y_vars=y_vars, cached=True))

                results.append(func(inputs, *args) if executor is None else executor.submit(func, inputs, *args))

        if executor is not None:
            wait(results, timeout=None, return_when=ALL_COMPLETED)
            results = [future.result() for future in results]

        for coeff, interp_pred in zip(coeffs, results):
            for var, arr in interp_pred.items():
                if outputs.get(var) is None:
                    outputs[str(var)] = coeff * arr
                else:
                    outputs[str(var)] += coeff * arr

        return format_outputs(outputs, loop_shape)

    def hessian(self, *args, **kwargs):
        """Alias for `Component.gradient(*args, derivative='second', **kwargs)`."""
        return self.gradient(*args, derivative='second', **kwargs)

    def model_kwarg_requested(self, kwarg_name: str) -> bool:
        """Return whether the underlying component model requested this `kwarg_name`. Special kwargs include:

        - `output_path` — a save directory created by `amisc` will be passed to the model for saving model output files.
        - `alpha` — a tuple or list of model fidelity indices will be passed to the model to adjust fidelity.
        - `input_vars` — a list of `Variable` objects will be passed to the model for input variable information.
        - `output_vars` — a list of `Variable` objects will be passed to the model for output variable information.

        :param kwarg_name: the argument to check for in the underlying component model's function signature kwargs
        :returns: whether the component model requests this `kwarg` argument
        """
        signature = inspect.signature(self.model)
        for param in signature.parameters.values():
            if param.name == kwarg_name and param.default != param.empty:
                return True
        return False

    def set_logger(self, log_file: str | Path = None, stdout: bool = None, logger: logging.Logger = None,
                   level: int = logging.INFO):
        """Set a new `logging.Logger` object.

        :param log_file: log to file (if provided)
        :param stdout: whether to connect the logger to console (defaults to whatever is currently set or False)
        :param logger: the logging object to use (if None, then a new logger is created; this will override
                       the `log_file` and `stdout` arguments if set)
        :param level: the logging level to set (default is `logging.INFO`)
        """
        if stdout is None:
            stdout = False
            if self._logger is not None:
                for handler in self._logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        stdout = True
                        break
        self._logger = logger or get_logger(self.name, log_file=log_file, stdout=stdout, level=level)

    def update_model(self, new_model: callable = None, model_kwargs: dict = None, **kwargs):
        """Update the underlying component model or its kwargs."""
        if new_model is not None:
            self.model = new_model
        new_kwargs = self.model_kwargs.data
        new_kwargs.update(model_kwargs or {})
        new_kwargs.update(kwargs)
        self.model_kwargs = new_kwargs

    def get_cost(self, alpha: MultiIndex, beta: MultiIndex) -> int:
        """Return the total cost (i.e. number of model evaluations) required to add $(\\alpha, \\beta)$ to the
        MISC approximation.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        :returns: the total number of model evaluations required for adding this multi-index to the MISC approximation
        """
        try:
            return self.misc_costs[alpha, beta]
        except Exception:
            return 0

    def get_model_timestamps(self):
        """Return a tuple with the (start, end) timestamps for the most recent call to `call_model`. This
        is useful for tracking the duration of model evaluations. Will return (None, None) if no model has been called.
        """
        if self._model_start_time < 0 or self._model_end_time < 0:
            return None, None
        else:
            return self._model_start_time, self._model_end_time

    @staticmethod
    def is_downward_closed(indices: IndexSet) -> bool:
        """Return if a list of $(\\alpha, \\beta)$ multi-indices is downward-closed.

        MISC approximations require a downward-closed set in order to use the combination-technique formula for the
        coefficients (as implemented by `Component.update_misc_coeff()`).

        !!! Example
            The list `[( (0,), (0,) ), ( (1,), (0,) ), ( (1,), (1,) )]` is downward-closed. You can visualize this as
            building a stack of cubes: in order to place a cube, all adjacent cubes must be present (does the logo
            make sense now?).

        :param indices: `IndexSet` of (`alpha`, `beta`) multi-indices
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

    def clear(self):
        """Clear the component of all training data, index sets, and MISC states."""
        self.active_set.clear()
        self.candidate_set.clear()
        self.misc_states.clear()
        self.misc_costs.clear()
        self.misc_coeff_train.clear()
        self.misc_coeff_test.clear()
        self.model_costs.clear()
        self.model_evals.clear()
        self.training_data.clear()
        self._model_start_time = -1.0
        self._model_end_time = -1.0
        self.clear_cache()

    def serialize(self, keep_yaml_objects: bool = False, serialize_args: dict[str, tuple] = None,
                  serialize_kwargs: dict[str: dict] = None) -> dict:
        """Convert to a `dict` with only standard Python types as fields and values.

        :param keep_yaml_objects: whether to keep `Variable` or other yaml serializable objects instead of
                                  also serializing them (default is False)
        :param serialize_args: additional arguments to pass to the `serialize` method of each `Component` attribute;
                               specify as a `dict` of attribute names to tuple of arguments to pass
        :param serialize_kwargs: additional keyword arguments to pass to the `serialize` method of each
                                 `Component` attribute
        :returns: a `dict` representation of the `Component` object
        """
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
                elif key in ['data_fidelity', 'surrogate_fidelity', 'model_fidelity']:
                    if len(value) > 0:
                        d[key] = str(value)
                elif key in ['active_set', 'candidate_set']:
                    if len(value) > 0:
                        d[key] = value.serialize()
                elif key in ['misc_costs', 'misc_coeff_train', 'misc_coeff_test', 'misc_states']:
                    if len(value) > 0:
                        d[key] = value.serialize(keep_yaml_objects=keep_yaml_objects)
                elif key in ['model_costs']:
                    if len(value) > 0:
                        d[key] = {str(k): float(v) for k, v in value.items()}
                elif key in ['model_evals']:
                    if len(value) > 0:
                        d[key] = {str(k): int(v) for k, v in value.items()}
                elif key in ComponentSerializers.__annotations__.keys():
                    if key in ['training_data'] and not self.has_surrogate:
                        continue
                    else:
                        d[key] = value.serialize(*serialize_args.get(key, ()), **serialize_kwargs.get(key, {}))
                else:
                    d[key] = value
        return d

    @classmethod
    def deserialize(cls, serialized_data: dict, search_paths: list[str | Path] = None,
                    search_keys: list[str] = None) -> Component:
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
            # try to construct a component from a raw function (assume data fidelity is (2,) for each inspected input)
            return cls(serialized_data, data_fidelity=(2,) * len(_inspect_function(serialized_data)[0]))

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
