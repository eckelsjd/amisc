"""The `System` object is a framework for multidisciplinary models. It manages multiple single discipline component
models and the connections between them. It provides a top-level interface for constructing and evaluating surrogates.

Features:

- Manages multidisciplinary models in a graph data structure, supports feedforward and feedback connections
- Feedback connections are solved with a fixed-point iteration (FPI) nonlinear solver with anderson acceleration
- Top-level interface for training and using surrogates of each component model
- Adaptive experimental design for choosing training data efficiently
- Convenient testing, plotting, and performance metrics provided to assess quality of surrogates
- Detailed logging and traceback information
- Supports parallel or vectorized execution of component models
- Abstract and flexible interfacing with component models
- Easy serialization and deserialization to/from YAML files
- Supports approximating field quantities via compression

Includes:

- `TrainHistory` — a history of training iterations for the system surrogate
- `System` — the top-level object for managing multidisciplinary models
"""
# ruff: noqa: E702
from __future__ import annotations

import copy
import datetime
import functools
import logging
import os
import pickle
import random
import string
import time
import warnings
from collections import ChainMap, UserList, deque
from concurrent.futures import ALL_COMPLETED, Executor, wait
from datetime import timezone
from pathlib import Path
from typing import Annotated, Callable, ClassVar, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel, ConfigDict, Field, field_validator

from amisc.component import Component, IndexSet, MiscTree
from amisc.serialize import Serializable, _builtin
from amisc.typing import COORDS_STR_ID, LATENT_STR_ID, Dataset, MultiIndex, TrainIteration
from amisc.utils import (
    _combine_latent_arrays,
    constrained_lls,
    format_inputs,
    format_outputs,
    get_logger,
    relative_error,
    to_model_dataset,
    to_surrogate_dataset,
)
from amisc.variable import VariableList

__all__ = ['TrainHistory', 'System']


class TrainHistory(UserList, Serializable):
    """Stores the training history of a system surrogate as a list of `TrainIteration` objects."""

    def __init__(self, data: list = None):
        data = data or []
        super().__init__(self._validate_data(data))

    def serialize(self) -> list[dict]:
        """Return a list of each result in the history serialized to a `dict`."""
        ret_list = []
        for res in self:
            new_res = res.copy()
            new_res['alpha'] = str(res['alpha'])
            new_res['beta'] = str(res['beta'])
            ret_list.append(new_res)
        return ret_list

    @classmethod
    def deserialize(cls, serialized_data: list[dict]) -> TrainHistory:
        """Deserialize a list of `dict` objects into a `TrainHistory` object."""
        return TrainHistory(serialized_data)

    @classmethod
    def _validate_data(cls, data: list[dict]) -> list[TrainIteration]:
        return [cls._validate_item(item) for item in data]

    @classmethod
    def _validate_item(cls, item: dict):
        """Format a `TrainIteration` `dict` item before appending to the history."""
        item.setdefault('test_error', None)
        item.setdefault('overhead_s', 0.0)
        item.setdefault('model_s', 0.0)
        item['alpha'] = MultiIndex(item['alpha'])
        item['beta'] = MultiIndex(item['beta'])
        item['num_evals'] = int(item['num_evals'])
        item['added_cost'] = float(item['added_cost'])
        item['added_error'] = float(item['added_error'])
        item['overhead_s'] = float(item['overhead_s'])
        item['model_s'] = float(item['model_s'])
        return item

    def append(self, item: dict):
        super().append(self._validate_item(item))

    def __add__(self, other):
        other_list = other.data if isinstance(other, TrainHistory) else other
        return TrainHistory(data=self.data + other_list)

    def extend(self, items):
        super().extend([self._validate_item(item) for item in items])

    def insert(self, index, item):
        super().insert(index, self._validate_item(item))

    def __setitem__(self, key, value):
        super().__setitem__(key, self._validate_item(value))

    def __eq__(self, other):
        """Two `TrainHistory` objects are equal if they have the same length and all items are equal, excluding nans."""
        if not isinstance(other, TrainHistory):
            return False
        if len(self) != len(other):
            return False
        for item_self, item_other in zip(self, other):
            for key in item_self:
                if key in item_other:
                    val_self = item_self[key]
                    val_other = item_other[key]
                    if isinstance(val_self, float) and isinstance(val_other, float):
                        if not (np.isnan(val_self) and np.isnan(val_other)) and val_self != val_other:
                            return False
                    elif isinstance(val_self, dict) and isinstance(val_other, dict):
                        for v, err in val_self.items():
                            if v in val_other:
                                err_other = val_other[v]
                                if not (np.isnan(err) and np.isnan(err_other)) and err != err_other:
                                    return False
                            else:
                                return False
                    elif val_self != val_other:
                        return False
                else:
                    return False
        return True


class _Converged:
    """Helper class to track which samples have converged during `System.predict()`."""
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.valid_idx = np.full(num_samples, True)          # All samples are valid by default
        self.converged_idx = np.full(num_samples, False)     # For FPI convergence

    def reset_convergence(self):
        self.converged_idx = np.full(self.num_samples, False)

    @property
    def curr_idx(self):
        return np.logical_and(self.valid_idx, ~self.converged_idx)


def _merge_shapes(target_shape, arr):
    """Helper to merge an array into the target shape."""
    shape1, shape2 = target_shape, arr.shape
    if len(shape2) > len(shape1):
        shape1, shape2 = shape2, shape1
    result = []
    for i in range(len(shape1)):
        if i < len(shape2):
            if shape1[i] == 1:
                result.append(shape2[i])
            elif shape2[i] == 1:
                result.append(shape1[i])
            else:
                result.append(shape1[i])
        else:
            result.append(1)
    arr = arr.reshape(tuple(result))
    return np.broadcast_to(arr, target_shape).copy()


class System(BaseModel, Serializable):
    """
    Multidisciplinary (MD) surrogate framework top-level class. Construct a `System` from a list of
    `Component` models.

    !!! Example
        ```python
        def f1(x):
            y = x ** 2
            return y
        def f2(y):
            z = y + 1
            return z

        system = System(f1, f2)
        ```

    A `System` object can saved/loaded from `.yml` files using the `!System` yaml tag.

    :ivar name: the name of the system
    :ivar components: list of `Component` models that make up the MD system
    :ivar train_history: history of training iterations for the system surrogate (filled in during training)

    :ivar _root_dir: root directory where all surrogate build products are saved to file
    :ivar _logger: logger object for the system
    """
    yaml_tag: ClassVar[str] = u'!System'
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, validate_default=True,
                              extra='allow')

    name: Annotated[str, Field(default_factory=lambda: "System_" + "".join(random.choices(string.digits, k=3)))]
    components: Callable | Component | list[Callable | Component]
    train_history: list[dict] | TrainHistory = TrainHistory()
    amisc_version: str = None

    _root_dir: Optional[str]
    _logger: Optional[logging.Logger] = None

    def __init__(self, /, *args, components=None, root_dir=None, **kwargs):
        """Construct a `System` object from a list of `Component` models in `*args` or `components`. If
        a `root_dir` is provided, then a new directory will be created under `root_dir` with the name
        `amisc_{timestamp}`. This directory will be used to save all build products and log files.

        :param components: list of `Component` models that make up the MD system
        :param root_dir: root directory where all surrogate build products are saved to file (optional)
        """
        if components is None:
            components = []
            for a in args:
                if isinstance(a, Component) or callable(a):
                    components.append(a)
                else:
                    try:
                        components.extend(a)
                    except TypeError as e:
                        raise ValueError(f"Invalid component: {a}") from e

        import amisc
        amisc_version = kwargs.pop('amisc_version', amisc.__version__)
        super().__init__(components=components, amisc_version=amisc_version, **kwargs)
        self.root_dir = root_dir

    def __repr__(self):
        s = f'---- {self.name} ----\n'
        s += f'amisc version: {self.amisc_version}\n'
        s += f'Refinement level: {self.refine_level}\n'
        s += f'Components: {", ".join([comp.name for comp in self.components])}\n'
        s += f'Inputs:     {", ".join([var.name for var in self.inputs()])}\n'
        s += f'Outputs:    {", ".join([var.name for var in self.outputs()])}'
        return s

    def __str__(self):
        return self.__repr__()

    @field_validator('components')
    @classmethod
    def _validate_components(cls, comps) -> list[Component]:
        if not isinstance(comps, list):
            comps = [comps]
        comps = [Component.deserialize(c) for c in comps]

        # Merge all variables to avoid name conflicts
        merged_vars = VariableList.merge(*[comp.inputs for comp in comps], *[comp.outputs for comp in comps])
        for comp in comps:
            comp.inputs.update({var.name: var for var in merged_vars.values() if var in comp.inputs})
            comp.outputs.update({var.name: var for var in merged_vars.values() if var in comp.outputs})

        return comps

    @field_validator('train_history')
    @classmethod
    def _validate_train_history(cls, history) -> TrainHistory:
        if isinstance(history, TrainHistory):
            return history
        else:
            return TrainHistory.deserialize(history)

    def graph(self) -> nx.DiGraph:
        """Build a directed graph of the system components based on their input-output relationships."""
        graph = nx.DiGraph()
        model_deps = {}
        for comp in self.components:
            graph.add_node(comp.name)
            for output in comp.outputs:
                model_deps[output] = comp.name
        for comp in self.components:
            for in_var in comp.inputs:
                if in_var in model_deps:
                    graph.add_edge(model_deps[in_var], comp.name)

        return graph

    def _save_on_error(func):
        """Gracefully exit and save the `System` object on any errors."""
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except:
                if self.root_dir is not None:
                    self.save_to_file(f'{self.name}_error.yml')
                self.logger.critical(f'An error occurred during execution of "{func.__name__}". Saving '
                                     f'System object to {self.name}_error.yml', exc_info=True)
                self.logger.info(f'Final system surrogate on exit: \n {self}')
                raise
        return wrap
    _save_on_error = staticmethod(_save_on_error)

    def insert_components(self, components: list | Callable | Component):
        """Insert new components into the system."""
        components = components if isinstance(components, list) else [components]
        self.components = self.components + components

    def swap_component(self, old_component: str | Component, new_component: Callable | Component):
        """Replace an old component with a new component."""
        old_name = old_component if isinstance(old_component, str) else old_component.name
        comps = [comp if comp.name != old_name else new_component for comp in self.components]
        self.components = comps

    def remove_component(self, component: str | Component):
        """Remove a component from the system."""
        comp_name = component if isinstance(component, str) else component.name
        self.components = [comp for comp in self.components if comp.name != comp_name]

    def inputs(self) -> VariableList:
        """Collect all inputs from each component in the `System` and combine them into a
        single [`VariableList`][amisc.variable.VariableList] object, excluding variables that are also outputs of
        any component.

        :returns: A [`VariableList`][amisc.variable.VariableList] containing all inputs from the components.
        """
        all_inputs = ChainMap(*[comp.inputs for comp in self.components])
        return VariableList({k: all_inputs[k] for k in all_inputs.keys() - self.outputs().keys()})

    def outputs(self) -> VariableList:
        """Collect all outputs from each component in the `System` and combine them into a
        single [`VariableList`][amisc.variable.VariableList] object.

        :returns: A [`VariableList`][amisc.variable.VariableList] containing all outputs from the components.
        """
        return VariableList({k: v for k, v in ChainMap(*[comp.outputs for comp in self.components]).items()})

    def coupling_variables(self) -> VariableList:
        """Collect all coupling variables from each component in the `System` and combine them into a
        single [`VariableList`][amisc.variable.VariableList] object.

        :returns: A [`VariableList`][amisc.variable.VariableList] containing all coupling variables from the components.
        """
        all_outputs = self.outputs()
        return VariableList({k: all_outputs[k] for k in (all_outputs.keys() &
                             ChainMap(*[comp.inputs for comp in self.components]).keys())})

    def variables(self):
        """Iterator over all variables in the system (inputs and outputs)."""
        yield from ChainMap(self.inputs(), self.outputs()).values()

    @property
    def refine_level(self) -> int:
        """The total number of training iterations."""
        return len(self.train_history)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, logger: logging.Logger):
        self._logger = logger
        for comp in self.components:
            comp.logger = logger

    @staticmethod
    def timestamp() -> str:
        """Return a UTC timestamp string in the isoformat `YYYY-MM-DDTHH.MM.SS`."""
        return datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')

    @property
    def root_dir(self):
        """Return the root directory of the surrogate (if available), otherwise `None`."""
        return Path(self._root_dir) if self._root_dir is not None else None

    @root_dir.setter
    def root_dir(self, root_dir: str | Path):
        """Set the root directory for all build products. If `root_dir` is `None`, then no products will be saved.
        Otherwise, log files, model outputs, surrogate files, etc. will be saved under this directory.

        !!! Note "`amisc` root directory"
            If `root_dir` is not `None`, then a new directory will be created under `root_dir` with the name
            `amisc_{timestamp}`. This directory will be used to save all build products. If `root_dir` matches the
            `amisc_*` format, then it will be used directly.

        :param root_dir: the root directory for all build products
        """
        if root_dir is not None:
            parts = Path(root_dir).resolve().parts
            if parts[-1].startswith('amisc_'):
                self._root_dir = Path(root_dir).resolve().as_posix()
                if not self.root_dir.is_dir():
                    os.mkdir(self.root_dir)
            else:
                root_dir = Path(root_dir) / ('amisc_' + self.timestamp())
                os.mkdir(root_dir)
                self._root_dir = Path(root_dir).resolve().as_posix()

            log_file = None
            if not (pth := self.root_dir / 'surrogates').is_dir():
                os.mkdir(pth)
            if not (pth := self.root_dir / 'components').is_dir():
                os.mkdir(pth)
            for comp in self.components:
                if comp.model_kwarg_requested('output_path'):
                    if not (comp_pth := pth / comp.name).is_dir():
                        os.mkdir(comp_pth)
            for f in os.listdir(self.root_dir):
                if f.endswith('.log'):
                    log_file = (self.root_dir / f).resolve().as_posix()
                    break
            if log_file is None:
                log_file = (self.root_dir / f'amisc_{self.timestamp()}.log').resolve().as_posix()
            self.set_logger(log_file=log_file)

        else:
            self._root_dir = None
            self.set_logger(log_file=None)

    def set_logger(self, log_file: str | Path | bool = None, stdout: bool = None, logger: logging.Logger = None,
                   level: int = logging.INFO):
        """Set a new `logging.Logger` object.

        :param log_file: log to this file if str or Path (defaults to whatever is currently set or empty);
                         set `False` to remove file logging or set `True` to create a default log file in the root dir
        :param stdout: whether to connect the logger to console (defaults to whatever is currently set or `False`)
        :param logger: the logging object to use (this will override the `log_file` and `stdout` arguments if set);
                       if `None`, then a new logger is created according to `log_file` and `stdout`
        :param level: the logging level to set the logger to (defaults to `logging.INFO`)
        """
        # Decide whether to use stdout
        if stdout is None:
            stdout = False
            if self._logger is not None:
                for handler in self._logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        stdout = True
                        break

        # Decide what log_file to use (if any)
        if log_file is True:
            log_file = pth / f'amisc_{self.timestamp()}.log' if (pth := self.root_dir) is not None else (
                f'amisc_{self.timestamp()}.log')
        elif log_file is None:
            if self._logger is not None:
                for handler in self._logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        log_file = handler.baseFilename
                        break
        elif log_file is False:
            log_file = None

        self._logger = logger or get_logger(self.name, log_file=log_file, stdout=stdout, level=level)

        for comp in self.components:
            comp.set_logger(log_file=log_file, stdout=stdout, logger=logger, level=level)

    def sample_inputs(self, size: tuple | int,
                      component: str = 'System',
                      normalize: bool = True,
                      use_pdf: bool | str | list[str] = False,
                      include: str | list[str] = None,
                      exclude: str | list[str] = None,
                      nominal: dict[str, float] = None) -> Dataset:
        """Return samples of the inputs according to provided options. Will return samples in the
        normalized/compressed space of the surrogate by default. See [`to_model_dataset`][amisc.utils.to_model_dataset]
        to convert the samples to be usable by the true model directly.

        :param size: tuple or integer specifying shape or number of samples to obtain
        :param component: which component to sample inputs for (defaults to full system exogenous inputs)
        :param normalize: whether to normalize the samples (defaults to True)
        :param use_pdf: whether to sample from variable pdfs (defaults to False, which will instead sample from the
                        variable domain bounds). If a string or list of strings is provided, then only those variables
                        or variable categories will be sampled using their pdfs.
        :param include: a list of variable or variable categories to include in the sampling. Defaults to using all
                        input variables.
        :param exclude: a list of variable or variable categories to exclude from the sampling. Empty by default.
        :param nominal: `dict(var_id=value)` of nominal values for params with relative uncertainty. Specify nominal
                        values as unnormalized (will be normalized if `normalize=True`)
        :returns: `dict` of `(*size,)` samples for each selected input variable
        """
        size = (size, ) if isinstance(size, int) else size
        nominal = nominal or dict()
        inputs = self.inputs() if component == 'System' else self[component].inputs
        if include is None:
            include = []
        if not isinstance(include, list):
            include = [include]
        if exclude is None:
            exclude = []
        if not isinstance(exclude, list):
            exclude = [exclude]
        if isinstance(use_pdf, str):
            use_pdf = [use_pdf]

        selected_inputs = []
        for var in inputs:
            if len(include) == 0 or var.name in include or var.category in include:
                if var.name not in exclude and var.category not in exclude:
                    selected_inputs.append(var)

        samples = {}
        for var in selected_inputs:
            # Sample from latent variable domains for field quantities
            if var.compression is not None:
                latent = var.sample_domain(size)
                for i in range(latent.shape[-1]):
                    samples[f'{var.name}{LATENT_STR_ID}{i}'] = latent[..., i]

            # Sample scalars normally
            else:
                if (domain := var.get_domain()) is None:
                    raise RuntimeError(f"Trying to sample variable '{var}' with empty domain. Please set a domain "
                                       f"for this variable. Samples outside the provided domain will be rejected.")
                lb, ub = domain
                pdf = (var.name in use_pdf or var.category in use_pdf) if isinstance(use_pdf, list) else use_pdf
                nom = nominal.get(var.name, None)

                x_sample = var.sample(size, nominal=nom) if pdf else var.sample_domain(size)
                good_idx = (x_sample < ub) & (x_sample > lb)
                num_reject = np.sum(~good_idx)

                while num_reject > 0:
                    new_sample = var.sample((num_reject,), nominal=nom) if pdf else var.sample_domain((num_reject,))
                    x_sample[~good_idx] = new_sample
                    good_idx = (x_sample < ub) & (x_sample > lb)
                    num_reject = np.sum(~good_idx)

                samples[var.name] = var.normalize(x_sample) if normalize else x_sample

        return samples

    def simulate_fit(self):
        """Loop back through training history and simulate each iteration. Will yield the internal data structures
        of each `Component` surrogate after each iteration of training (without needing to call `fit()` or any
        of the underlying models). This might be useful, for example, for computing the surrogate predictions on
        a new test set or viewing cumulative training costs.

        !!! Example
            Say you have a new test set: `(new_xtest, new_ytest)`, and you want to compute the accuracy of the
            surrogate fit at each iteration of the training history:

            ```python
            for train_iter, active_sets, candidate_sets, misc_coeff_train, misc_coeff_test in system.simulate_fit():
                # Do something with the surrogate data structures
                new_ysurr = system.predict(new_xtest, index_set=active_sets, misc_coeff=misc_coeff_train)
                train_error = relative_error(new_ysurr, new_ytest)
            ```

        :return: a generator of the active index sets, candidate index sets, and MISC coefficients
                 of each component model at each iteration of the training history
        """
        # "Simulated" data structures for each component
        active_sets = {comp.name: IndexSet() for comp in self.components}       # active index sets for each component
        candidate_sets = {comp.name: IndexSet() for comp in self.components}    # candidate sets for each component
        misc_coeff_train = {comp.name: MiscTree() for comp in self.components}  # MISC coeff for active sets
        misc_coeff_test = {comp.name: MiscTree() for comp in self.components}   # MISC coeff for active + candidate sets

        for train_result in self.train_history:
            # The selected refinement component and indices
            comp_star = train_result['component']
            alpha_star = train_result['alpha']
            beta_star = train_result['beta']
            comp = self[comp_star]

            # Get forward neighbors for the selected index
            neighbors = comp._neighbors(alpha_star, beta_star, active_set=active_sets[comp_star], forward=True)

            # "Activate" the index in the simulated data structure
            s = set()
            s.add((alpha_star, beta_star))
            comp.update_misc_coeff(IndexSet(s), index_set=active_sets[comp_star],
                                   misc_coeff=misc_coeff_train[comp_star])

            if (alpha_star, beta_star) in candidate_sets[comp_star]:
                candidate_sets[comp_star].remove((alpha_star, beta_star))
            else:
                # Only for initial index which didn't come from the candidate set
                comp.update_misc_coeff(IndexSet(s), index_set=active_sets[comp_star].union(candidate_sets[comp_star]),
                                       misc_coeff=misc_coeff_test[comp_star])
            active_sets[comp_star].update(s)

            comp.update_misc_coeff(neighbors, index_set=active_sets[comp_star].union(candidate_sets[comp_star]),
                                   misc_coeff=misc_coeff_test[comp_star])  # neighbors will only ever pass here once
            candidate_sets[comp_star].update(neighbors)

            # Caller can now do whatever they want as if the system surrogate were at this training iteration
            # See the "index_set" and "misc_coeff" overrides for `System.predict()` for example
            yield train_result, active_sets, candidate_sets, misc_coeff_train, misc_coeff_test

    def add_output(self):
        """Add an output variable retroactively to a component surrogate. User should provide a callable that
        takes a save path and extracts the model output data for given training point/location.
        """
        # TODO
        # Loop back through the surrogate training history
        # Simulate activate_index and extract the model output from file rather than calling the model
        # Update all interpolator states
        raise NotImplementedError

    @_save_on_error
    def fit(self, targets: list = None,
            num_refine: int = 100,
            max_iter: int = 20,
            max_tol: float = 1e-3,
            runtime_hr: float = 1.,
            estimate_bounds: bool = False,
            update_bounds: bool = True,
            test_set: tuple | str | Path = None,
            start_test_check: int = None,
            save_interval: int = 0,
            plot_interval: int = 1,
            cache_interval: int = 0,
            executor: Executor = None,
            weight_fcns: dict[str, callable] | Literal['pdf'] | None = 'pdf'):
        """Train the system surrogate adaptively by iterative refinement until an end condition is met.

        :param targets: list of system output variables to focus refinement on, use all outputs if not specified
        :param num_refine: number of input samples to compute error indicators on
        :param max_iter: the maximum number of refinement steps to take
        :param max_tol: the max allowable value in relative L2 error to achieve
        :param runtime_hr: the threshold wall clock time (hr) at which to stop further refinement (will go
                           until all models finish the current iteration)
        :param estimate_bounds: whether to estimate bounds for the coupling variables; will only try to estimate from
                                the `test_set` if provided (defaults to `True`). Otherwise, you should manually
                                provide domains for all coupling variables.
        :param update_bounds: whether to continuously update coupling variable bounds during refinement
        :param test_set: `tuple` of `(xtest, ytest)` to show convergence of surrogate to the true model. The test set
                         inputs and outputs are specified as `dicts` of `np.ndarrays` with keys corresponding to the
                         variable names. Can also pass a path to a `.pkl` file that has the test set data as
                         {'test_set': (xtest, ytest)}.
        :param start_test_check: the iteration to start checking the test set error (defaults to the number
                                 of components); surrogate evaluation isn't useful during initialization so you
                                 should at least allow one iteration per component before checking test set error
        :param save_interval: number of refinement steps between each progress save, none if 0; `System.root_dir`
                              must be specified to save to file
        :param plot_interval: how often to plot the error indicator and test set error (defaults to every iteration);
                              will only plot and save to file if a root directory is set
        :param cache_interval: how often to cache component data in order to speed up future training iterations (at
                               the cost of additional memory usage); defaults to 0 (no caching)
        :param executor: a `concurrent.futures.Executor` object to parallelize model evaluations (optional, but
                         recommended for expensive models)
        :param weight_fcns: a `dict` of weight functions to apply to each input variable for training data selection;
                            defaults to using the pdf of each variable. If None, then no weighting is applied.
        """
        start_test_check = start_test_check or sum([1 for _ in self.components if _.has_surrogate])
        targets = targets or self.outputs()
        xtest, ytest = self._get_test_set(test_set)
        max_iter = self.refine_level + max_iter

        # Estimate bounds from test set if provided (override current bounds if they are set)
        if estimate_bounds:
            if ytest is not None:
                y_samples = to_surrogate_dataset(ytest, self.outputs(), del_fields=True)[0]  # normalize/compress
                _combine_latent_arrays(y_samples)
                coupling_vars = {k: v for k, v in self.coupling_variables().items() if k in y_samples}
                y_min, y_max = {}, {}
                for var in coupling_vars.values():
                    y_min[var] = np.nanmin(y_samples[var], axis=0)
                    y_max[var] = np.nanmax(y_samples[var], axis=0)
                    if var.compression is not None:
                        new_domain = list(zip(y_min[var].tolist(), y_max[var].tolist()))
                        var.update_domain(new_domain, override=True)
                    else:
                        new_domain = (float(y_min[var]), float(y_max[var]))
                        var.update_domain(var.denormalize(new_domain), override=True)
                del y_samples
            else:
                self.logger.warning('Could not estimate bounds for coupling variables: no test set provided. '
                                    'Make sure you manually provide (good) coupling variable domains.')

        # Track convergence progress on the error indicator and test set (plot to file)
        if self.root_dir is not None:
            err_record = [res['added_error'] for res in self.train_history]
            err_fig, err_ax = plt.subplots(figsize=(6, 5), layout='tight')

            if xtest is not None and ytest is not None:
                num_plot = min(len(targets), 3)
                test_record = np.full((self.refine_level, num_plot), np.nan)
                t_fig, t_ax = plt.subplots(1, num_plot, figsize=(3.5 * num_plot, 4), layout='tight', squeeze=False,
                                           sharey='row')
                for j, res in enumerate(self.train_history):
                    for i, var in enumerate(targets[:num_plot]):
                        if (perf := res.get('test_error')) is not None:
                            test_record[j, i] = perf[var]

        total_overhead = 0.0
        total_model_wall_time = 0.0
        t_start = time.time()
        while True:
            # Adaptive refinement step
            t_iter_start = time.time()
            train_result = self.refine(targets=targets, num_refine=num_refine, update_bounds=update_bounds,
                                       executor=executor, weight_fcns=weight_fcns)
            if train_result['component'] is None:
                self._print_title_str('Termination criteria reached: No candidates left to refine')
                break

            # Keep track of algorithmic overhead (before and after call_model for this iteration)
            m_start, m_end = self[train_result['component']].get_model_timestamps()  # Start and end of call_model
            if m_start is not None and m_end is not None:
                train_result['overhead_s'] = (m_start - t_iter_start) + (time.time() - m_end)
                train_result['model_s'] = m_end - m_start
            else:
                train_result['overhead_s'] = time.time() - t_iter_start
                train_result['model_s'] = 0.0
            total_overhead += train_result['overhead_s']
            total_model_wall_time += train_result['model_s']

            curr_error = train_result['added_error']

            # Plot progress of error indicator
            if self.root_dir is not None:
                err_record.append(curr_error)

                if plot_interval > 0 and self.refine_level % plot_interval == 0:
                    err_ax.clear(); err_ax.set_yscale('log'); err_ax.grid()
                    err_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    err_ax.plot(err_record, '-k')
                    err_ax.set_xlabel('Iteration'); err_ax.set_ylabel('Relative error indicator')
                    err_fig.savefig(str(Path(self.root_dir) / 'error_indicator.pdf'), format='pdf', bbox_inches='tight')

            # Save performance on a test set
            if xtest is not None and ytest is not None:
                # don't compute if components are uninitialized
                perf = self.test_set_performance(xtest, ytest) if self.refine_level + 1 >= start_test_check else (
                    {str(var): np.nan for var in ytest if COORDS_STR_ID not in var})
                train_result['test_error'] = perf.copy()

                if self.root_dir is not None:
                    test_record = np.vstack((test_record, np.array([perf[var] for var in targets[:num_plot]])))

                    if plot_interval > 0 and self.refine_level % plot_interval == 0:
                        for i in range(num_plot):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                t_ax[0, i].clear(); t_ax[0, i].set_yscale('log'); t_ax[0, i].grid()
                            t_ax[0, i].xaxis.set_major_locator(MaxNLocator(integer=True))
                            t_ax[0, i].plot(test_record[:, i], '-k')
                            t_ax[0, i].set_title(self.outputs()[targets[i]].get_tex(units=True))
                            t_ax[0, i].set_xlabel('Iteration')
                            t_ax[0, i].set_ylabel('Test set relative error' if i==0 else '')
                        t_fig.savefig(str(Path(self.root_dir) / 'test_set_error.pdf'),format='pdf',bbox_inches='tight')

            self.train_history.append(train_result)

            if self.root_dir is not None and save_interval > 0 and self.refine_level % save_interval == 0:
                iter_name = f'{self.name}_iter{self.refine_level}'
                if not (pth := self.root_dir / 'surrogates' / iter_name).is_dir():
                    os.mkdir(pth)
                self.save_to_file(f'{iter_name}.yml', save_dir=pth)  # Save to an iteration-specific directory

            if cache_interval > 0 and self.refine_level % cache_interval == 0:
                for comp in self.components:
                    comp.cache()

            # Check all end conditions
            if self.refine_level >= max_iter:
                self._print_title_str(f'Termination criteria reached: Max iteration {self.refine_level}/{max_iter}')
                break
            if curr_error < max_tol:
                self._print_title_str(f'Termination criteria reached: relative error {curr_error} < tol {max_tol}')
                break
            if ((time.time() - t_start) / 3600.0) >= runtime_hr:
                t_end = time.time()
                actual = datetime.timedelta(seconds=t_end - t_start)
                target = datetime.timedelta(seconds=runtime_hr * 3600)
                train_surplus = ((t_end - t_start) - runtime_hr * 3600) / 3600
                self._print_title_str(f'Termination criteria reached: runtime {str(actual)} > {str(target)}')
                self.logger.info(f'Surplus wall time: {train_surplus:.3f}/{runtime_hr:.3f} hours '
                                 f'(+{100 * train_surplus / runtime_hr:.2f}%)')
                break

        self.logger.info(f'Model evaluation algorithm efficiency: '
                         f'{100 * total_model_wall_time / (total_model_wall_time + total_overhead):.2f}%')

        if self.root_dir is not None:
            iter_name = f'{self.name}_iter{self.refine_level}'
            if not (pth := self.root_dir / 'surrogates' / iter_name).is_dir():
                os.mkdir(pth)
            self.save_to_file(f'{iter_name}.yml', save_dir=pth)

            if xtest is not None and ytest is not None:
                self._save_test_set((xtest, ytest))

        self.logger.info(f'Final system surrogate: \n {self}')

    def test_set_performance(self, xtest: Dataset, ytest: Dataset, index_set='test') -> Dataset:
        """Compute the relative L2 error on a test set for the given target output variables.

        :param xtest: `dict` of test set input samples   (unnormalized)
        :param ytest: `dict` of test set output samples  (unnormalized)
        :param index_set: index set to use for prediction (defaults to 'train')
        :returns: `dict` of relative L2 errors for each target output variable
        """
        targets = [var for var in ytest.keys() if COORDS_STR_ID not in var and var in self.outputs()]
        coords = {var: ytest[var] for var in ytest if COORDS_STR_ID in var}
        xtest = to_surrogate_dataset(xtest, self.inputs(), del_fields=True)[0]
        ysurr = self.predict(xtest, index_set=index_set, targets=targets)
        ysurr = to_model_dataset(ysurr, self.outputs(), del_latent=True, **coords)[0]
        perf = {}
        for var in targets:
            # Handle relative error for object arrays (field qtys)
            ytest_obj = np.issubdtype(ytest[var].dtype, np.object_)
            ysurr_obj = np.issubdtype(ysurr[var].dtype, np.object_)
            if ytest_obj or ysurr_obj:
                _iterable = np.ndindex(ytest[var].shape) if ytest_obj else np.ndindex(ysurr[var].shape)
                num, den = [], []
                for index in _iterable:
                    pred, targ = ysurr[var][index], ytest[var][index]
                    num.append((pred - targ)**2)
                    den.append(targ ** 2)
                perf[var] = float(np.sqrt(sum([np.sum(n) for n in num]) / sum([np.sum(d) for d in den])))
            else:
                perf[var] = float(relative_error(ysurr[var], ytest[var]))

        return perf

    def refine(self, targets: list = None, num_refine: int = 100, update_bounds: bool = True, executor: Executor = None,
               weight_fcns: dict[str, callable] | Literal['pdf'] | None = 'pdf') -> TrainIteration:
        """Perform a single adaptive refinement step on the system surrogate.

        :param targets: list of system output variables to focus refinement on, use all outputs if not specified
        :param num_refine: number of input samples to compute error indicators on
        :param update_bounds: whether to continuously update coupling variable bounds during refinement
        :param executor: a `concurrent.futures.Executor` object to parallelize model evaluations
        :param weight_fcns: weight functions for choosing new training data for each input variable; defaults to
                            the PDFs of each variable. If None, then no weighting is applied.
        :returns: `dict` of the refinement results indicating the chosen component and candidate index
        """
        self._print_title_str(f'Refining system surrogate: iteration {self.refine_level + 1}')
        targets = targets or self.outputs()

        # Check for uninitialized components and refine those first
        for comp in self.components:
            if len(comp.active_set) == 0 and comp.has_surrogate:
                alpha_star = (0,) * len(comp.model_fidelity)
                beta_star = (0,) * len(comp.max_beta)
                self.logger.info(f"Initializing component {comp.name}: adding {(alpha_star, beta_star)} to active set")
                model_dir = (pth / 'components' / comp.name) if (pth := self.root_dir) is not None else None
                comp.activate_index(alpha_star, beta_star, model_dir=model_dir, executor=executor,
                                    weight_fcns=weight_fcns)
                num_evals = comp.get_cost(alpha_star, beta_star)
                cost_star = max(1., comp.model_costs.get(alpha_star, 1.) * num_evals)  # Cpu time (s)
                err_star = np.nan
                return {'component': comp.name, 'alpha': alpha_star, 'beta': beta_star, 'num_evals': int(num_evals),
                        'added_cost': float(cost_star), 'added_error': float(err_star)}

        # Compute entire integrated-surrogate on a random test set for global system QoI error estimation
        x_samples = self.sample_inputs(num_refine)
        y_curr = self.predict(x_samples, index_set='train', targets=targets)
        _combine_latent_arrays(y_curr)
        coupling_vars = {k: v for k, v in self.coupling_variables().items() if k in y_curr}

        y_min, y_max = None, None
        if update_bounds:
            y_min = {var: np.nanmin(y_curr[var], axis=0, keepdims=True) for var in coupling_vars}  # (1, ydim)
            y_max = {var: np.nanmax(y_curr[var], axis=0, keepdims=True) for var in coupling_vars}  # (1, ydim)

        # Find the candidate surrogate with the largest error indicator
        error_max, error_indicator = -np.inf, -np.inf
        comp_star, alpha_star, beta_star, err_star, cost_star = None, None, None, -np.inf, 0
        for comp in self.components:
            if not comp.has_surrogate:  # Skip analytic models that don't need a surrogate
                continue

            self.logger.info(f"Estimating error for component '{comp.name}'...")

            if len(comp.candidate_set) > 0:
                candidates = list(comp.candidate_set)
                if executor is None:
                    ret = [self.predict(x_samples, targets=targets, index_set={comp.name: {(alpha, beta)}},
                                        incremental={comp.name: True})
                           for alpha, beta in candidates]
                else:
                    temp_buffer = self._remove_unpickleable()
                    futures = [executor.submit(self.predict, x_samples, targets=targets,
                                               index_set={comp.name: {(alpha, beta)}}, incremental={comp.name: True})
                               for alpha, beta in candidates]
                    wait(futures, timeout=None, return_when=ALL_COMPLETED)
                    ret = [f.result() for f in futures]
                    self._restore_unpickleable(temp_buffer)

                for i, y_cand in enumerate(ret):
                    alpha, beta = candidates[i]
                    _combine_latent_arrays(y_cand)
                    error = {}
                    for var, arr in y_cand.items():
                        if var in targets:
                            error[var] = relative_error(arr, y_curr[var], skip_nan=True)

                        if update_bounds and var in coupling_vars:
                            y_min[var] = np.nanmin(np.concatenate((y_min[var], arr), axis=0), axis=0, keepdims=True)
                            y_max[var] = np.nanmax(np.concatenate((y_max[var], arr), axis=0), axis=0, keepdims=True)

                    delta_error = np.nanmax([np.nanmax(error[var]) for var in error])  # Max error over all target QoIs
                    num_evals = comp.get_cost(alpha, beta)
                    delta_work = max(1., comp.model_costs.get(alpha, 1.) * num_evals)  # Cpu time (s)
                    error_indicator = delta_error / delta_work

                    self.logger.info(f"Candidate multi-index: {(alpha, beta)}. Relative error: {delta_error}. "
                                     f"Error indicator: {error_indicator}.")

                    if error_indicator > error_max:
                        error_max = error_indicator
                        comp_star, alpha_star, beta_star, err_star, cost_star = (
                            comp.name, alpha, beta, delta_error, delta_work)
            else:
                self.logger.info(f"Component '{comp.name}' has no available candidates left!")

        # Update all coupling variable ranges
        if update_bounds:
            for var in coupling_vars.values():
                if np.all(~np.isnan(y_min[var])) and np.all(~np.isnan(y_max[var])):
                    if var.compression is not None:
                        new_domain = list(zip(np.squeeze(y_min[var], axis=0).tolist(),
                                              np.squeeze(y_max[var], axis=0).tolist()))
                        var.update_domain(new_domain)
                    else:
                        new_domain = (y_min[var][0], y_max[var][0])
                        var.update_domain(var.denormalize(new_domain))  # bds will be in norm space from predict() call

        # Add the chosen multi-index to the chosen component
        if comp_star is not None:
            self.logger.info(f"Candidate multi-index {(alpha_star, beta_star)} chosen for component '{comp_star}'.")
            model_dir = (pth / 'components' / comp_star) if (pth := self.root_dir) is not None else None
            self[comp_star].activate_index(alpha_star, beta_star, model_dir=model_dir, executor=executor,
                                           weight_fcns=weight_fcns)
            num_evals = self[comp_star].get_cost(alpha_star, beta_star)
        else:
            self.logger.info(f"No candidates left for refinement, iteration: {self.refine_level}")
            num_evals = 0

        # Return the results of the refinement step
        return {'component': comp_star, 'alpha': alpha_star, 'beta': beta_star, 'num_evals': int(num_evals),
                'added_cost': float(cost_star), 'added_error': float(err_star)}

    def predict(self, x: dict | Dataset,
                max_fpi_iter: int = 100,
                anderson_mem: int = 10,
                fpi_tol: float = 1e-10,
                use_model: str | tuple | dict = None,
                model_dir: str | Path = None,
                verbose: bool = False,
                index_set: dict[str: IndexSet | Literal['train', 'test']] = 'test',
                misc_coeff: dict[str: MiscTree] = None,
                normalized_inputs: bool = True,
                incremental: dict[str, bool] = False,
                targets: list[str] = None,
                executor: Executor = None,
                var_shape: dict[str, tuple] = None) -> Dataset:
        """Evaluate the system surrogate at inputs `x`. Return `y = system(x)`.

        !!! Warning "Computing the true model with feedback loops"
            You can use this function to predict outputs for your MD system using the full-order models rather than the
            surrogate, by specifying `use_model`. This is convenient because the `System` manages all the
            coupled information flow between models automatically. However, it is *highly* recommended to not use
            the full model if your system contains feedback loops. The FPI nonlinear solver would be infeasible using
            anything more computationally demanding than the surrogate.

        :param x: `dict` of input samples for each variable in the system
        :param max_fpi_iter: the limit on convergence for the fixed-point iteration routine
        :param anderson_mem: hyperparameter for tuning the convergence of FPI with anderson acceleration
        :param fpi_tol: tolerance limit for convergence of fixed-point iteration
        :param use_model: 'best'=highest-fidelity, 'worst'=lowest-fidelity, tuple=specific fidelity, None=surrogate,
                           specify a `dict` of the above to assign different model fidelities for diff components
        :param model_dir: directory to save model outputs if `use_model` is specified
        :param verbose: whether to print out iteration progress during execution
        :param index_set: `dict(comp=[indices])` to override the active set for a component, defaults to using the
                          `test` set for every component. Can also specify `train` for any component or a valid
                          `IndexSet` object. If `incremental` is specified, will be overwritten with `train`.
        :param misc_coeff: `dict(comp=MiscTree)` to override the default coefficients for a component, passes through
                           along with `index_set` and `incremental` to `comp.predict()`.
        :param normalized_inputs: true if the passed inputs are compressed/normalized for surrogate evaluation
                                  (default), such as inputs returned by `sample_inputs`. Set to `False` if you are
                                  passing inputs as the true models would expect them instead (i.e. not normalized).
        :param incremental: whether to add `index_set` to the current active set for each component (temporarily);
                            this will set `index_set='train'` for all other components (since incremental will
                            augment the "training" active sets, not the "testing" candidate sets)
        :param targets: list of output variables to return, defaults to returning all system outputs
        :param executor: a `concurrent.futures.Executor` object to parallelize model evaluations
        :param var_shape: (Optional) `dict` of shapes for field quantity inputs in `x` -- you would only specify this
                          if passing field qtys directly to the models (i.e. not using `sample_inputs`)
        :returns: `dict` of output variables - the surrogate approximation of the system outputs (or the true model)
        """
        # Format inputs and allocate space
        var_shape = var_shape or {}
        x, loop_shape = format_inputs(x, var_shape=var_shape)  # {'x': (N, *var_shape)}
        y = {}
        all_inputs = ChainMap(x, y)   # track all inputs (including coupling vars in y)
        N = int(np.prod(loop_shape))
        t1 = 0
        output_dir = None
        norm_status = {var: normalized_inputs for var in x}  # keep track of whether inputs are normalized or not
        graph = self.graph()

        # Keep track of what outputs are computed
        is_computed = {}
        for var in (targets or self.outputs()):
            if (v := self.outputs().get(var, None)) is not None:
                if v.compression is not None:
                    for field in v.compression.fields:
                        is_computed[field] = False
                else:
                    is_computed[var] = False

        def _set_default(struct: dict, default=None):
            """Helper to set a default value for each component key in a `dict`. Ensures all components have a value."""
            if struct is not None:
                if not isinstance(struct, dict):
                    struct = {node: struct for node in graph.nodes}  # use same for each component
            else:
                struct = {node: default for node in graph.nodes}
            return {node: struct.get(node, default) for node in graph.nodes}

        # Ensure use_model, index_set, and incremental are specified for each component model
        use_model = _set_default(use_model, None)
        incremental = _set_default(incremental, False)  # default to train if incremental anywhere
        index_set = _set_default(index_set, 'train' if any([incremental[node] for node in graph.nodes]) else 'test')
        misc_coeff = _set_default(misc_coeff, None)

        samples = _Converged(N)  # track convergence of samples

        def _gather_comp_inputs(comp, coupling=None):
            """Helper to gather inputs for a component, making sure they are normalized correctly. Any coupling
            variables passed in will be used in preference over `all_inputs`.
            """
            # Will access but not modify: all_inputs, use_model, norm_status
            field_coords = {}
            comp_input = {}
            coupling = coupling or {}

            # Take coupling variables as a priority
            comp_input.update({var: np.copy(arr[samples.curr_idx, ...]) for var, arr in
                               coupling.items() if str(var).split(LATENT_STR_ID)[0] in comp.inputs})
            # Gather all other inputs
            for var, arr in all_inputs.items():
                var_id = str(var).split(LATENT_STR_ID)[0]
                if var_id in comp.inputs and var not in coupling:
                    comp_input[var] = np.copy(arr[samples.curr_idx, ...])

            # Gather field coordinates
            for var in comp.inputs:
                coords_str = f'{var}{COORDS_STR_ID}'
                if (coords := all_inputs.get(coords_str)) is not None:
                    field_coords[coords_str] = coords[samples.curr_idx, ...]
                elif (coords := comp.model_kwargs.get(coords_str)) is not None:
                    field_coords[coords_str] = coords

            # Gather extra fields (will never be in coupling since field couplings should always be latent coeff)
            for var in comp.inputs:
                if var not in comp_input and var.compression is not None:
                    for field in var.compression.fields:
                        if field in all_inputs:
                            comp_input[field] = np.copy(all_inputs[field][samples.curr_idx, ...])

            call_model = use_model.get(comp.name, None) is not None

            # Make sure we format all inputs for model evaluation (i.e. denormalize)
            if call_model:
                norm_inputs = {var: arr for var, arr in comp_input.items() if norm_status[var]}
                if len(norm_inputs) > 0:
                    denorm_inputs, fc = to_model_dataset(norm_inputs, comp.inputs, del_latent=True, **field_coords)
                    for var in norm_inputs:
                        del comp_input[var]
                    field_coords.update(fc)
                    comp_input.update(denorm_inputs)

            # Otherwise, make sure we format inputs for surrogate evaluation (i.e. normalize)
            else:
                denorm_inputs = {var: arr for var, arr in comp_input.items() if not norm_status[var]}
                if len(denorm_inputs) > 0:
                    norm_inputs, _ = to_surrogate_dataset(denorm_inputs, comp.inputs, del_fields=True, **field_coords)

                    for var in denorm_inputs:
                        del comp_input[var]
                    comp_input.update(norm_inputs)

            return comp_input, field_coords, call_model

        # Convert system into DAG by grouping strongly-connected-components
        dag = nx.condensation(graph)

        # Compute component models in topological order
        for supernode in nx.topological_sort(dag):
            if np.all(list(is_computed.values())):
                break  # Exit early if all selected return qois are computed

            scc = [n for n in dag.nodes[supernode]['members']]
            samples.reset_convergence()

            # Compute single component feedforward output (no FPI needed)
            if len(scc) == 1:
                if verbose:
                    self.logger.info(f"Running component '{scc[0]}'...")
                    t1 = time.time()

                # Gather inputs
                comp = self[scc[0]]
                comp_input, field_coords, call_model = _gather_comp_inputs(comp)

                # Compute outputs
                if model_dir is not None:
                    output_dir = Path(model_dir) / scc[0]
                    if not output_dir.exists():
                        os.mkdir(output_dir)
                comp_output = comp.predict(comp_input, use_model=use_model.get(scc[0]), model_dir=output_dir,
                                           index_set=index_set.get(scc[0]), incremental=incremental.get(scc[0]),
                                           misc_coeff=misc_coeff.get(scc[0]), executor=executor, **field_coords)

                for var, arr in comp_output.items():
                    if var == 'errors':
                        if y.get(var) is None:
                            y.setdefault(var, np.full((N,), None, dtype=object))
                        global_indices = np.arange(N)[samples.curr_idx]

                        for local_idx, err_info in arr.items():
                            global_idx = int(global_indices[local_idx])
                            err_info['index'] = global_idx
                            y[var][global_idx] = err_info
                        continue

                    is_numeric = np.issubdtype(arr.dtype, np.number)
                    if is_numeric:  # for scalars or vectorized field quantities
                        output_shape = arr.shape[1:]
                        if y.get(var) is None:
                            y.setdefault(var, np.full((N, *output_shape), np.nan))
                        y[var][samples.curr_idx, ...] = arr

                    else:  # for fields returned as object arrays
                        if y.get(var) is None:
                            y.setdefault(var, np.full((N,), None, dtype=object))
                        y[var][samples.curr_idx] = arr

                # Update valid indices and status for component outputs
                for var in comp_output:
                    if str(var).split(LATENT_STR_ID)[0] in comp.outputs:
                        is_numeric = np.issubdtype(y[var].dtype, np.number)
                        new_valid = ~np.any(np.isnan(y[var]), axis=tuple(range(1, y[var].ndim))) if is_numeric else (
                            [False if arr is None else ~np.any(np.isnan(arr)) for i, arr in enumerate(y[var])]
                        )
                        samples.valid_idx = np.logical_and(samples.valid_idx, new_valid)

                        is_computed[str(var).split(LATENT_STR_ID)[0]] = True
                        norm_status[var] = not call_model

                if verbose:
                    self.logger.info(f"Component '{scc[0]}' completed. Runtime: {time.time() - t1} s")

            # Handle FPI for SCCs with more than one component
            else:
                # Set the initial guess for all coupling vars (middle of domain)
                scc_inputs = ChainMap(*[self[comp].inputs for comp in scc])
                scc_outputs = ChainMap(*[self[comp].outputs for comp in scc])
                coupling_vars = [scc_inputs.get(var) for var in (scc_inputs.keys() - x.keys()) if var in scc_outputs]
                coupling_prev = {}
                for var in coupling_vars:
                    if (domain := var.get_domain()) is None:
                        raise RuntimeError(f"Coupling variable '{var}' has an empty domain. All coupling variables "
                                           f"require a domain for the fixed-point iteration (FPI) solver.")

                    if isinstance(domain, list):  # Latent coefficients are the coupling variables
                        for i, d in enumerate(domain):
                            lb, ub = d
                            coupling_prev[f'{var.name}{LATENT_STR_ID}{i}'] = np.broadcast_to((lb + ub) / 2, (N,)).copy()
                            norm_status[f'{var.name}{LATENT_STR_ID}{i}'] = True
                    else:
                        lb, ub = var.normalize(domain)
                        shape = (N,) + (1,) * len(var_shape.get(var, ()))
                        coupling_prev[var] = np.broadcast_to((lb + ub) / 2, shape).copy()
                        norm_status[var] = True

                residual_hist = deque(maxlen=anderson_mem)
                coupling_hist = deque(maxlen=anderson_mem)

                def _end_conditions_met():
                    """Helper to compute residual, update history, and check end conditions."""
                    residual = {}
                    converged_idx = np.full(N, True)
                    for var in coupling_prev:
                        residual[var] = y[var] - coupling_prev[var]
                        var_conv = np.all(np.abs(residual[var]) <= fpi_tol, axis=tuple(range(1, residual[var].ndim)))
                        converged_idx = np.logical_and(converged_idx, var_conv)
                        samples.valid_idx = np.logical_and(samples.valid_idx, ~np.isnan(coupling_prev[var]))
                    samples.converged_idx = np.logical_or(samples.converged_idx, converged_idx)

                    for var in coupling_prev:
                        coupling_prev[var][samples.curr_idx, ...] = y[var][samples.curr_idx, ...]
                    residual_hist.append(copy.deepcopy(residual))
                    coupling_hist.append(copy.deepcopy(coupling_prev))

                    if int(np.sum(samples.curr_idx)) == 0:
                        if verbose:
                            self.logger.info(f'FPI converged for SCC {scc} in {k} iterations with tol '
                                             f'{fpi_tol}. Final time: {time.time() - t1} s')
                        return True

                    max_error = np.max([np.max(np.abs(res[samples.curr_idx, ...])) for res in residual.values()])
                    if verbose:
                        self.logger.info(f'FPI iter: {k}. Max residual: {max_error}. Time: {time.time() - t1} s')

                    if k >= max_fpi_iter:
                        self.logger.warning(f'FPI did not converge in {max_fpi_iter} iterations for SCC {scc}: '
                                            f'{max_error} > tol {fpi_tol}. Some samples will be returned as NaN.')
                        for var in coupling_prev:
                            y[var][~samples.converged_idx, ...] = np.nan
                        samples.valid_idx = np.logical_and(samples.valid_idx, samples.converged_idx)
                        return True
                    else:
                        return False

                # Main FPI loop
                if verbose:
                    self.logger.info(f"Initializing FPI for SCC {scc} ...")
                    t1 = time.time()
                k = 0
                while True:
                    for node in scc:
                        # Gather inputs from exogenous and coupling sources
                        comp = self[node]
                        comp_input, kwds, call_model = _gather_comp_inputs(comp, coupling=coupling_prev)

                        # Compute outputs (just don't do this FPI with expensive real models, please..)
                        comp_output = comp.predict(comp_input, use_model=use_model.get(node), model_dir=None,
                                                   index_set=index_set.get(node), incremental=incremental.get(node),
                                                   misc_coeff=misc_coeff.get(node), executor=executor, **kwds)

                        for var, arr in comp_output.items():
                            if var == 'errors':
                                if y.get(var) is None:
                                    y.setdefault(var, np.full((N,), None, dtype=object))
                                global_indices = np.arange(N)[samples.curr_idx]

                                for local_idx, err_info in arr.items():
                                    global_idx = int(global_indices[local_idx])
                                    err_info['index'] = global_idx
                                    y[var][global_idx] = err_info
                                continue

                            if np.issubdtype(arr.dtype, np.number):  # scalars and vectorized field quantities
                                output_shape = arr.shape[1:]
                                if y.get(var) is not None:
                                    if output_shape != y.get(var).shape[1:]:
                                        y[var] = _merge_shapes((N, *output_shape), y[var])
                                else:
                                    y.setdefault(var, np.full((N, *output_shape), np.nan))
                                y[var][samples.curr_idx, ...] = arr
                            else:  # fields returned as object arrays
                                if y.get(var) is None:
                                    y.setdefault(var, np.full((N,), None, dtype=object))
                                y[var][samples.curr_idx] = arr

                            if str(var).split(LATENT_STR_ID)[0] in comp.outputs:
                                norm_status[var] = not call_model
                                is_computed[str(var).split(LATENT_STR_ID)[0]] = True

                    # Compute residual and check end conditions
                    if _end_conditions_met():
                        break

                    # Skip anderson acceleration on first iteration
                    if k == 0:
                        k += 1
                        continue

                    # Iterate with anderson acceleration (only iterate on samples that are not yet converged)
                    N_curr = int(np.sum(samples.curr_idx))
                    mk = len(residual_hist)  # Max of anderson mem
                    var_shapes = []
                    xdims = []
                    for var in coupling_prev:
                        shape = coupling_prev[var].shape[1:]
                        var_shapes.append(shape)
                        xdims.append(int(np.prod(shape)))
                    N_couple = int(np.sum(xdims))
                    res_snap = np.empty((N_curr, N_couple, mk))       # Shortened snapshot of residual history
                    coupling_snap = np.empty((N_curr, N_couple, mk))  # Shortened snapshot of coupling history
                    for i, (coupling_iter, residual_iter) in enumerate(zip(coupling_hist, residual_hist)):
                        start_idx = 0
                        for j, var in enumerate(coupling_prev):
                            end_idx = start_idx + xdims[j]
                            coupling_snap[:, start_idx:end_idx, i] = coupling_iter[var][samples.curr_idx, ...].reshape((N_curr, -1))  # noqa: E501
                            res_snap[:, start_idx:end_idx, i] = residual_iter[var][samples.curr_idx, ...].reshape((N_curr, -1))  # noqa: E501
                            start_idx = end_idx
                    C = np.ones((N_curr, 1, mk))
                    b = np.zeros((N_curr, N_couple, 1))
                    d = np.ones((N_curr, 1, 1))
                    alpha = np.expand_dims(constrained_lls(res_snap, b, C, d), axis=-3)   # (..., 1, mk, 1)
                    coupling_new = np.squeeze(coupling_snap[:, :, np.newaxis, :] @ alpha, axis=(-1, -2))
                    start_idx = 0
                    for j, var in enumerate(coupling_prev):
                        end_idx = start_idx + xdims[j]
                        coupling_prev[var][samples.curr_idx, ...] = coupling_new[:, start_idx:end_idx].reshape((N_curr, *var_shapes[j]))  # noqa: E501
                        start_idx = end_idx
                    k += 1

        # Return all component outputs; samples that didn't converge during FPI are left as np.nan
        return format_outputs(y, loop_shape)

    def __call__(self, *args, **kwargs):
        """Convenience wrapper to allow calling as `ret = System(x)`."""
        return self.predict(*args, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, System):
            return False
        return (self.components == other.components and
                self.name == other.name and
                self.train_history == other.train_history)

    def __getitem__(self, component: str) -> Component:
        """Convenience method to get a `Component` object from the `System`.

        :param component: the name of the component to get
        :returns: the `Component` object
        """
        return self.get_component(component)

    def get_component(self, comp_name: str) -> Component:
        """Return the `Component` object for this component.

        :param comp_name: name of the component to return
        :raises KeyError: if the component does not exist
        :returns: the `Component` object
        """
        if comp_name.lower() == 'system':
            return self
        else:
            for comp in self.components:
                if comp.name == comp_name:
                    return comp
            raise KeyError(f"Component '{comp_name}' not found in system.")

    def _print_title_str(self, title_str: str):
        """Log an important message."""
        self.logger.info('-' * int(len(title_str)/2) + title_str + '-' * int(len(title_str)/2))

    def _remove_unpickleable(self) -> dict:
        """Remove and return unpickleable attributes before pickling (just the logger)."""
        stdout = False
        log_file = None
        if self._logger is not None:
            for handler in self._logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    stdout = True
                    break
            for handler in self._logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file = handler.baseFilename
                    break

        buffer = {'log_stdout': stdout, 'log_file': log_file}
        self.logger = None
        return buffer

    def _restore_unpickleable(self, buffer: dict):
        """Restore the unpickleable attributes from `buffer` after unpickling."""
        self.set_logger(log_file=buffer.get('log_file', None), stdout=buffer.get('log_stdout', None))

    def _get_test_set(self, test_set: str | Path | tuple = None) -> tuple:
        """Try to load a test set from the root directory if it exists."""
        if isinstance(test_set, tuple):
            return test_set  # (xtest, ytest)
        else:
            ret = (None, None)
            if test_set is not None:
                test_set = Path(test_set)
            elif self.root_dir is not None:
                test_set = self.root_dir / 'test_set.pkl'

            if test_set is not None:
                if test_set.exists():
                    with open(test_set, 'rb') as fd:
                        data = pickle.load(fd)
                    ret = data['test_set']

            return ret

    def _save_test_set(self, test_set: tuple = None):
        """Save the test set to the root directory if possible."""
        if self.root_dir is not None and test_set is not None:
            test_file = self.root_dir / 'test_set.pkl'
            if not test_file.exists():
                with open(test_file, 'wb') as fd:
                    pickle.dump({'test_set': test_set}, fd)

    def save_to_file(self, filename: str, save_dir: str | Path = None, dumper=None):
        """Save surrogate to file. Defaults to `root/surrogates/filename.yml` with the default yaml encoder.

        :param filename: the name of the save file
        :param save_dir: the directory to save the file to (defaults to `root/surrogates` or `cwd()`)
        :param dumper: the encoder to use (defaults to the `amisc` yaml encoder)
        """
        from amisc import YamlLoader
        encoder = dumper or YamlLoader
        save_dir = save_dir or self.root_dir or Path.cwd()
        if Path(save_dir) == self.root_dir:
            save_dir = self.root_dir / 'surrogates'
        encoder.dump(self, Path(save_dir) / filename)

    @staticmethod
    def load_from_file(filename: str | Path, root_dir: str | Path = None, loader=None):
        """Load surrogate from file. Defaults to yaml loading. Tries to infer `amisc` directory structure.

        :param filename: the name of the load file
        :param root_dir: set this as the surrogate's root directory (will try to load from `amisc_` fmt by default)
        :param loader: the encoder to use (defaults to the `amisc` yaml encoder)
        """
        from amisc import YamlLoader
        encoder = loader or YamlLoader
        system = encoder.load(filename)
        root_dir = root_dir or system.root_dir

        # Try to infer amisc_root/surrogates/iter/filename structure
        if root_dir is None:
            parts = Path(filename).resolve().parts
            if len(parts) > 1 and parts[-2].startswith('amisc_'):
                root_dir = Path(filename).resolve().parent
            elif len(parts) > 2 and parts[-3].startswith('amisc_'):
                root_dir = Path(filename).resolve().parent.parent
            elif len(parts) > 3 and parts[-4].startswith('amisc_'):
                root_dir = Path(filename).resolve().parent.parent.parent

        system.root_dir = root_dir
        return system

    def clear(self):
        """Clear all surrogate model data and reset the system."""
        for comp in self.components:
            comp.clear()
        self.train_history.clear()

    def plot_slice(self, inputs: list[str] = None,
                   outputs: list[str] = None,
                   num_steps: int = 20,
                   show_surr: bool = True,
                   show_model: str | tuple | list = None,
                   save_dir: str | Path = None,
                   executor: Executor = None,
                   nominal: dict[str: float] = None,
                   random_walk: bool = False,
                   from_file: str | Path = None,
                   subplot_size_in: float = 3.):
        """Helper function to plot 1d slices of the surrogate and/or model outputs over the inputs. A single
        "slice" works by smoothly stepping from the lower bound of an input to its upper bound, while holding all other
        inputs constant at their nominal values (or smoothly varying them if `random_walk=True`).
        This function is useful for visualizing the behavior of the system surrogate and/or model(s) over a
        single input variable at a time.

        :param inputs: list of input variables to take 1d slices of (defaults to first 3 in `System.inputs`)
        :param outputs: list of model output variables to plot 1d slices of (defaults to first 3 in `System.outputs`)
        :param num_steps: the number of points to take in the 1d slice for each input variable; this amounts to a total
                          of `num_steps*len(inputs)` model/surrogate evaluations
        :param show_surr: whether to show the surrogate prediction
        :param show_model: also compute and plot model predictions, `list` of ['best', 'worst', tuple(alpha), etc.]
        :param save_dir: base directory to save model outputs and plots (if specified)
        :param executor: a `concurrent.futures.Executor` object to parallelize model or surrogate evaluations
        :param nominal: `dict` of `var->nominal` to use as constant values for all non-sliced variables (use
                        unnormalized values only; use `var_LATENT0` to specify nominal latent values)
        :param random_walk: whether to slice in a random d-dimensional direction instead of holding all non-slice
                            variables const at `nominal`
        :param from_file: path to a `.pkl` file to load a saved slice from disk
        :param subplot_size_in: side length size of each square subplot in inches
        :returns: `fig, ax` with `len(inputs)` by `len(outputs)` subplots
        """
        # Manage loading important quantities from file (if provided)
        input_slices, output_slices_model, output_slices_surr = None, None, None
        if from_file is not None:
            with open(Path(from_file), 'rb') as fd:
                slice_data = pickle.load(fd)
                inputs = slice_data['inputs']           # Must use same input slices as save file
                show_model = slice_data['show_model']   # Must use same model data as save file
                outputs = slice_data.get('outputs') if outputs is None else outputs
                input_slices = slice_data['input_slices']
                save_dir = None  # Don't run or save any models if loading from file

        # Set default values (take up to the first 3 inputs by default)
        all_inputs = self.inputs()
        all_outputs = self.outputs()
        rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        if save_dir is not None:
            os.mkdir(Path(save_dir) / f'slice_{rand_id}')
        if nominal is None:
            nominal = dict()
        inputs = all_inputs[:3] if inputs is None else inputs
        outputs = all_outputs[:3] if outputs is None else outputs

        if show_model is not None and not isinstance(show_model, list):
            show_model = [show_model]

        # Handle field quantities (directly use latent variables or only the first one)
        for i, var in enumerate(list(inputs)):
            if LATENT_STR_ID not in str(var) and all_inputs[var].compression is not None:
                inputs[i] = f'{var}{LATENT_STR_ID}0'
        for i, var in enumerate(list(outputs)):
            if LATENT_STR_ID not in str(var) and all_outputs[var].compression is not None:
                outputs[i] = f'{var}{LATENT_STR_ID}0'

        bds = all_inputs.get_domains()
        xlabels = [all_inputs[var].get_tex(units=False) if LATENT_STR_ID not in str(var) else
                   all_inputs[str(var).split(LATENT_STR_ID)[0]].get_tex(units=False) +
                   f' (latent {str(var).split(LATENT_STR_ID)[1]})' for var in inputs]

        ylabels = [all_outputs[var].get_tex(units=False) if LATENT_STR_ID not in str(var) else
                   all_outputs[str(var).split(LATENT_STR_ID)[0]].get_tex(units=False) +
                   f' (latent {str(var).split(LATENT_STR_ID)[1]})' for var in outputs]

        # Construct slices of model inputs (if not provided)
        if input_slices is None:
            input_slices = {}  # Each input variable with shape (num_steps, num_slice)
            for i in range(len(inputs)):
                if random_walk:
                    # Make a random straight-line walk across d-cube
                    r0 = self.sample_inputs((1,), use_pdf=False)
                    rf = self.sample_inputs((1,), use_pdf=False)

                    for var, bd in bds.items():
                        if var == inputs[i]:
                            r0[var] = np.atleast_1d(bd[0])          # Start slice at this lower bound
                            rf[var] = np.atleast_1d(bd[1])          # Slice up to this upper bound

                        step_size = (rf[var] - r0[var]) / (num_steps - 1)
                        arr = r0[var] + step_size * np.arange(num_steps)

                        input_slices[var] = arr[..., np.newaxis] if input_slices.get(var) is None else (
                            np.concatenate((input_slices[var], arr[..., np.newaxis]), axis=-1))
                else:
                    # Otherwise, only slice one variable
                    for var, bd in bds.items():
                        nom = nominal.get(var, np.mean(bd)) if LATENT_STR_ID in str(var) else (
                            all_inputs[var].normalize(nominal.get(var, all_inputs[var].get_nominal())))
                        arr = np.linspace(bd[0], bd[1], num_steps) if var == inputs[i] else np.full(num_steps, nom)

                        input_slices[var] = arr[..., np.newaxis] if input_slices.get(var) is None else (
                            np.concatenate((input_slices[var], arr[..., np.newaxis]), axis=-1))

        # Walk through each model that is requested by show_model
        if show_model is not None:
            if from_file is not None:
                output_slices_model = slice_data['output_slices_model']
            else:
                output_slices_model = list()
                for model in show_model:
                    output_dir = None
                    if save_dir is not None:
                        output_dir = (Path(save_dir) / f'slice_{rand_id}' /
                                      str(model).replace('{', '').replace('}', '').replace(':', '=').replace("'", ''))
                        os.mkdir(output_dir)
                    output_slices_model.append(self.predict(input_slices, use_model=model, model_dir=output_dir,
                                                            executor=executor))
        if show_surr:
            output_slices_surr = self.predict(input_slices, executor=executor) \
                if from_file is None else slice_data['output_slices_surr']

        # Make len(outputs) by len(inputs) grid of subplots
        fig, axs = plt.subplots(len(outputs), len(inputs), sharex='col', sharey='row', squeeze=False)
        for i, output_var in enumerate(outputs):
            for j, input_var in enumerate(inputs):
                ax = axs[i, j]
                x = input_slices[input_var][:, j]

                if show_model is not None:
                    c = np.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1]]) if len(show_model) <= 2 else (
                        plt.get_cmap('jet')(np.linspace(0, 1, len(show_model))))
                    for k in range(len(show_model)):
                        model_str = (str(show_model[k]).replace('{', '').replace('}', '')
                                     .replace(':', '=').replace("'", ''))
                        model_ret = to_surrogate_dataset(output_slices_model[k], all_outputs)[0]
                        y_model = model_ret[output_var][:, j]
                        label = {'best': 'High-fidelity' if len(show_model) > 1 else 'Model',
                                 'worst': 'Low-fidelity'}.get(model_str, model_str)
                        ax.plot(x, y_model, ls='-', c=c[k, :], label=label)

                if show_surr:
                    y_surr = output_slices_surr[output_var][:, j]
                    ax.plot(x, y_surr, '--r', label='Surrogate')

                ax.set_xlabel(xlabels[j] if i == len(outputs) - 1 else '')
                ax.set_ylabel(ylabels[i] if j == 0 else '')
                if i == 0 and j == len(inputs) - 1:
                    ax.legend()
        fig.set_size_inches(subplot_size_in * len(inputs), subplot_size_in * len(outputs))
        fig.tight_layout()

        # Save results (unless we were already loading from a save file)
        if from_file is None and save_dir is not None:
            fname = f'in={",".join([str(v) for v in inputs])}_out={",".join([str(v) for v in outputs])}'
            fname = f'slice_rand{rand_id}_' + fname if random_walk else f'slice_nom{rand_id}_' + fname
            fdir = Path(save_dir) / f'slice_{rand_id}'
            fig.savefig(fdir / f'{fname}.pdf', bbox_inches='tight', format='pdf')
            save_dict = {'inputs': inputs, 'outputs': outputs, 'show_model': show_model, 'show_surr': show_surr,
                         'nominal': nominal, 'random_walk': random_walk, 'input_slices': input_slices,
                         'output_slices_model': output_slices_model, 'output_slices_surr': output_slices_surr}
            with open(fdir / f'{fname}.pkl', 'wb') as fd:
                pickle.dump(save_dict, fd)

        return fig, axs

    def get_allocation(self):
        """Get a breakdown of cost allocation during training.

        :returns: `cost_alloc, model_cost, overhead_cost, model_evals` - the cost allocation per model/fidelity,
                  the model evaluation cost per iteration (in s of CPU time), the algorithmic overhead cost per
                  iteration, and the total number of model evaluations at each training iteration
        """
        cost_alloc = dict()     # Cost allocation (cpu time in s) per node and model fidelity
        model_cost = []         # Cost of model evaluations (CPU time in s) per iteration
        overhead_cost = []      # Algorithm overhead costs (CPU time in s) per iteration
        model_evals = []        # Number of model evaluations at each training iteration

        prev_cands = {comp.name: IndexSet() for comp in self.components}  # empty candidate sets

        # Add cumulative training costs
        for train_res, active_sets, cand_sets, misc_coeff_train, misc_coeff_test in self.simulate_fit():
            comp = train_res['component']
            alpha = train_res['alpha']
            beta = train_res['beta']
            overhead = train_res['overhead_s']

            cost_alloc.setdefault(comp, dict())

            new_cands = cand_sets[comp].union({(alpha, beta)}) - prev_cands[comp]  # newly computed candidates

            iter_cost = 0.
            iter_eval = 0
            for alpha_new, beta_new in new_cands:
                cost_alloc[comp].setdefault(alpha_new, 0.)

                added_eval = self[comp].get_cost(alpha_new, beta_new)
                single_cost = self[comp].model_costs.get(alpha_new, 1.)

                iter_cost += added_eval * single_cost
                iter_eval += added_eval

                cost_alloc[comp][alpha_new] += added_eval * single_cost

            overhead_cost.append(overhead)
            model_cost.append(iter_cost)
            model_evals.append(iter_eval)
            prev_cands[comp] = cand_sets[comp].union({(alpha, beta)})

        return cost_alloc, np.atleast_1d(model_cost), np.atleast_1d(overhead_cost), np.atleast_1d(model_evals)

    def plot_allocation(self, cmap: str = 'Blues', text_bar_width: float = 0.06, arrow_bar_width: float = 0.02):
        """Plot bar charts showing cost allocation during training.

        !!! Warning "Beta feature"
            This has pretty good default settings, but it might look terrible for your use. Mostly provided here as
            a template for making cost allocation bar charts. Please feel free to copy and edit in your own code.

        :param cmap: the colormap string identifier for `plt`
        :param text_bar_width: the minimum total cost fraction above which a bar will print centered model fidelity text
        :param arrow_bar_width: the minimum total cost fraction above which a bar will try to print text with an arrow;
                                below this amount, the bar is too skinny and won't print any text
        :returns: `fig, ax`, Figure and Axes objects
        """
        # Get total cost
        cost_alloc, model_cost, _, _ = self.get_allocation()
        total_cost = np.cumsum(model_cost)[-1]

        # Remove nodes with cost=0 from alloc dicts (i.e. analytical models)
        remove_nodes = []
        for node, alpha_dict in cost_alloc.items():
            if len(alpha_dict) == 0:
                remove_nodes.append(node)
        for node in remove_nodes:
            del cost_alloc[node]

        # Bar chart showing cost allocation breakdown for MF system at final iteration
        fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
        width = 0.7
        x = np.arange(len(cost_alloc))
        xlabels = list(cost_alloc.keys())  # One bar for each component
        cmap = plt.get_cmap(cmap)

        for j, (node, alpha_dict) in enumerate(cost_alloc.items()):
            bottom = 0
            c_intervals = np.linspace(0, 1, len(alpha_dict))
            bars = [(alpha, cost, cost / total_cost) for alpha, cost in alpha_dict.items()]
            bars = sorted(bars, key=lambda ele: ele[2], reverse=True)
            for i, (alpha, cost, frac) in enumerate(bars):
                p = ax.bar(x[j], frac, width, color=cmap(c_intervals[i]), linewidth=1,
                           edgecolor=[0, 0, 0], bottom=bottom)
                bottom += frac
                num_evals = round(cost / self[node].model_costs.get(alpha, 1.))
                if frac > text_bar_width:
                    ax.bar_label(p, labels=[f'{alpha}, {num_evals}'], label_type='center')
                elif frac > arrow_bar_width:
                    xy = (x[j] + width / 2, bottom - frac / 2)  # Label smaller bars with a text off to the side
                    ax.annotate(f'{alpha}, {num_evals}', xy, xytext=(xy[0] + 0.2, xy[1]),
                                arrowprops={'arrowstyle': '->', 'linewidth': 1})
                else:
                    pass  # Don't label really small bars
        ax.set_xlabel('')
        ax.set_ylabel('Fraction of total cost')
        ax.set_xticks(x, xlabels)
        ax.set_xlim(left=-1, right=x[-1] + 1)

        if self.root_dir is not None:
            fig.savefig(Path(self.root_dir) / 'mf_allocation.pdf', bbox_inches='tight', format='pdf')

        return fig, ax

    def serialize(self, keep_components=False, serialize_args=None, serialize_kwargs=None) -> dict:
        """Convert to a `dict` with only standard Python types for fields.

        :param keep_components: whether to serialize the components as well (defaults to False)
        :param serialize_args: `dict` of arguments to pass to each component's serialize method
        :param serialize_kwargs: `dict` of keyword arguments to pass to each component's serialize method
        :returns: a `dict` representation of the `System` object
        """
        serialize_args = serialize_args or dict()
        serialize_kwargs = serialize_kwargs or dict()
        d = {}
        for key, value in self.__dict__.items():
            if value is not None and not key.startswith('_'):
                if key == 'components' and not keep_components:
                    d[key] = [comp.serialize(keep_yaml_objects=False,
                                             serialize_args=serialize_args.get(comp.name),
                                             serialize_kwargs=serialize_kwargs.get(comp.name))
                              for comp in value]
                elif key == 'train_history':
                    if len(value) > 0:
                        d[key] = value.serialize()
                else:
                    if not isinstance(value, _builtin):
                        self.logger.warning(f"Attribute '{key}' of type '{type(value)}' may not be a builtin "
                                            f"Python type. This may cause issues when saving/loading from file.")
                    d[key] = value

        for key, value in self.model_extra.items():
            if isinstance(value, _builtin):
                d[key] = value

        return d

    @classmethod
    def deserialize(cls, serialized_data: dict) -> System:
        """Construct a `System` object from serialized data."""
        return cls(**serialized_data)

    @staticmethod
    def _yaml_representer(dumper: yaml.Dumper, system: System):
        """Serialize a `System` object to a YAML representation."""
        return dumper.represent_mapping(System.yaml_tag, system.serialize(keep_components=True))

    @staticmethod
    def _yaml_constructor(loader: yaml.Loader, node):
        """Convert the `!SystemSurrogate` tag in yaml to a [`System`][amisc.system.System] object."""
        if isinstance(node, yaml.SequenceNode):
            return [ele if isinstance(ele, System) else System.deserialize(ele)
                    for ele in loader.construct_sequence(node, deep=True)]
        elif isinstance(node, yaml.MappingNode):
            return System.deserialize(loader.construct_mapping(node, deep=True))
        else:
            raise NotImplementedError(f'The "{System.yaml_tag}" yaml tag can only be used on a yaml sequence or '
                                      f'mapping, not a "{type(node)}".')
