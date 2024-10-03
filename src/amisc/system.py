"""The `SystemSurrogate` is a framework for multidisciplinary models. It manages multiple single discipline component
models and the connections between them. It provides a top-level interface for constructing and evaluating surrogates.

Features
--------
- Manages multidisciplinary models in a graph data structure, supports feedforward and feedback connections
- Feedback connections are solved with a fixed-point iteration (FPI) nonlinear solver
- FPI uses Anderson acceleration and surrogate evaluations for speed-up
- Top-level interface for training and using surrogates of each component model
- Adaptive experimental design for choosing training data efficiently
- Convenient testing, plotting, and performance metrics provided to assess quality of surrogates
- Detailed logging and traceback information
- Supports parallel execution with OpenMP and MPI protocols
- Abstract and flexible interfacing with component models

!!! Info "Model specification"
    Models are callable Python wrapper functions of the form `ret = model(x, *args, **kwargs)`, where `x` is an
    `np.ndarray` of model inputs (and `*args, **kwargs` allow passing any other required configurations for your model).
    The return value is a Python dictionary of the form `ret = {'y': y, 'files': files, 'cost': cost, etc.}`. In the
    return dictionary, you specify the raw model output `y` as an `np.ndarray` at a _minimum_. Optionally, you can
    specify paths to output files and the average model cost (in seconds of cpu time), and anything else you want. Your
    `model()` function can do anything it wants in order to go from `x` &rarr; `y`. Python has the flexibility to call
    virtually any external codes, or to implement the function natively with `numpy`.

!!! Info "Component specification"
    A component adds some extra configuration around a callable `model`. These configurations are defined in a Python
    dictionary, which we give the custom type `ComponentSpec`. At a bare _minimum_, you must specify a callable
    `model` and its connections to other models within the multidisciplinary system. The limiting case is a single
    component model, for which the configuration is simply `component = ComponentSpec(model)`.
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
from collections import ChainMap, deque
from concurrent.futures import Executor
from datetime import timezone
from pathlib import Path
from typing import ClassVar, Annotated, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from pydantic import BaseModel, ConfigDict, Field, field_validator
from uqtils import ax_default

from amisc.component import IndexSet, ComponentIO
from amisc.component import Component
from amisc.serialize import Serializable
from amisc.utils import get_logger, format_inputs, format_outputs
from amisc.variable import Variable, VariableList


class System(BaseModel, Serializable):
    yaml_tag: ClassVar[str] = u'!System'
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, validate_default=True,
                              extra='allow')

    components: Component | list[Component]
    name: Annotated[str, Field(default_factory=lambda: "System_" + "".join(random.choices(string.digits, k=3)))]
    train_history: dict = dict()  # TODO: custom type/validator?

    _graph: nx.DiGraph
    _root_dir: Optional[str]
    _logger: Optional[logging.Logger] = None
    _executor: Optional[Executor]

    def __init__(self, /, *args, components=None, executor=None, root_dir=None, **kwargs):
        if components is None:
            components = []
            for a in args:
                if isinstance(a, Component):
                    components.append(a)
                else:
                    try:
                        components.extend(a)
                    except TypeError as e:
                        raise ValueError(f"Invalid component: {a}") from e
        super().__init__(components=components, **kwargs)
        self.root_dir = root_dir
        self.executor = executor
        self.build_graph()

    def __repr__(self):
        s = f'---- {self.name} ----\n'
        s += f'Refinement: {self.refine_level}\n'
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
        return [Component.deserialize(c) for c in comps]

    def build_graph(self):
        """Build a directed graph of the system components based on their input-output relationships."""
        self._graph = nx.DiGraph()
        model_deps = {}
        for comp in self.components:
            self._graph.add_node(comp.name, component=comp)
            for output in comp.outputs:
                model_deps[output] = comp.name
        for comp in self.components:
            for in_var in comp.inputs:
                if in_var in model_deps:
                    self._graph.add_edge(model_deps[in_var], comp.name)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def insert_component(self, component: Component):
        """Insert a new component into the system."""
        self.components.append(component)
        self.build_graph()

    def swap_component(self, old_component: str | Component, new_component: Component):
        """Replace an old component with a new component."""
        old_name = old_component if isinstance(old_component, str) else old_component.name
        comp_names = [comp.name for comp in self.components]
        idx = comp_names.index(old_name)
        self.components[idx] = new_component
        self.build_graph()

    def remove_component(self, component: str | Component):
        """Remove a component from the system."""
        comp_name = component if isinstance(component, str) else component.name
        self.components = [comp for comp in self.components if comp.name != comp_name]
        self.build_graph()

    def inputs(self) -> VariableList:
        """Collect all inputs from each component in the `System` and combine them into a
        single [`VariableList`][amisc.variable.VariableList] object, excluding variables that are also outputs of
        any component.

        :returns: A [`VariableList`][amisc.variable.VariableList] containing all inputs from the components.
        """
        all_inputs = ChainMap(*[comp.inputs for comp in self.components])
        return VariableList({k: all_inputs[k] for k in all_inputs.keys() -
                             ChainMap(*[comp.outputs for comp in self.components]).keys()})

    def outputs(self) -> VariableList:
        """Collect all outputs from each component in the `System` and combine them into a
        single [`VariableList`][amisc.variable.VariableList] object, excluding variables that are also outputs of
        any component.

        :returns: A [`VariableList`][amisc.variable.VariableList] containing all inputs from the components.
        """
        return VariableList({k: v for k, v in ChainMap(*[comp.outputs for comp in self.components]).items()})

    @property
    def refine_level(self) -> int:
        return len(self.train_history)

    @property
    def executor(self) -> Executor:
        return self._executor

    @executor.setter
    def executor(self, executor: Executor):
        self._executor = executor
        for comp in self.components:
            comp.executor = executor

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @staticmethod
    def timestamp() -> str:
        return datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')

    @property
    def root_dir(self):
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
                if comp.model_arg_requested('output_path'):
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

        for comp in self.components:
            comp.set_logger(log_file=log_file, stdout=stdout, logger=logger)

    def sample_inputs(self, size: tuple | int, comp: str = 'System', use_pdf: bool = False, transform: bool = False,
                      nominal: dict[str: float] = None, constants: set[str] = None) -> dict | ComponentIO:
        """Return samples of the inputs according to provided options.

        :param size: tuple or integer specifying shape or number of samples to obtain
        :param comp: which component to sample inputs for (defaults to full system exogenous inputs)
        :param use_pdf: whether to sample from each variable's pdf, defaults to random samples over input domain instead
        :param transform: whether to sample from the transformed variable domain(s), defaults to the original domain
        :param nominal: `dict(var_id=value)` of nominal values for params with relative uncertainty, also can use
                        to specify constant values for a variable listed in `constants`
        :param constants: set of param types to hold constant while sampling (i.e. calibration, design, etc.),
                          can also put a `var_id` string in here to specify a single variable to hold constant
        :returns: `dict` of `(*size,)` samples for each input variable
        """
        size = (size, ) if isinstance(size, int) else size
        nominal = nominal or dict()
        constants = constants or set()
        inputs = self.inputs() if comp == 'System' else self[comp].inputs
        samples = {}
        for var in inputs:
            # Set a constant value for this variable
            if var.category in constants or var in constants:
                samples[var] = nominal.get(var, var.get_nominal(transform=transform))

            # Sample from this variable's pdf or randomly within its domain bounds (reject if outside bounds)
            else:
                lb, ub = var.get_domain(transform=transform)
                x_sample = var.sample(size, nominal=nominal.get(var, None), transform=transform) if use_pdf \
                    else var.sample_domain(size, transform=transform)
                good_idx = (x_sample < ub) & (x_sample > lb)
                num_reject = np.sum(~good_idx)

                while num_reject > 0:
                    new_sample = var.sample((num_reject,), nominal=nominal.get(var, None), transform=transform) \
                        if use_pdf else var.sample_domain((num_reject,), transform=transform)
                    x_sample[~good_idx] = new_sample
                    good_idx = (x_sample < ub) & (x_sample > lb)
                    num_reject = np.sum(~good_idx)

                samples[var] = x_sample

        return samples

    def predict(self, x: dict | ComponentIO, max_fpi_iter: int = 100, anderson_mem: int = 10, fpi_tol: float = 1e-10,
                use_model: str | tuple | dict = None, model_dir: str | Path = None, verbose: bool = False,
                training: bool = False, index_set: dict[str: IndexSet] = None, ret_outputs=None) -> dict | ComponentIO:
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
        :param training: whether to call the system surrogate in training or evaluation mode, ignored if `use_model`
        :param index_set: `dict(comp=[indices])` to override default index set for a component
        :param ret_outputs: list of output variables to return, defaults to returning all system outputs
        :returns: `dict` of output variables - the surrogate approximation of the system outputs (or the true model)
        """
        # TODO: argument to return latent coeff instead of reconstructed fields
        # Format inputs and allocate space
        x, loop_shape = format_inputs(x, var_shape={var: var.shape for var in self.inputs()})  # {'x': (N, ...)}
        y = {}
        all_inputs = ChainMap(x, y)
        N = int(np.prod(loop_shape))
        t1 = 0
        output_dir = None
        is_computed = {var: False for var in (ret_outputs or self.outputs())}

        class _Converged:
            """Store indices to track which samples have converged."""
            def __init__(self, num_samples):
                self.num_samples = num_samples
                self.valid_idx = np.full(num_samples, True)          # All samples are valid by default
                self.converged_idx = np.full(num_samples, False)     # For FPI convergence

            def reset_convergence(self):
                self.converged_idx = np.full(self.num_samples, False)

            @property
            def curr_idx(self):
                return np.logical_and(self.valid_idx, ~self.converged_idx)
        samples = _Converged(N)

        # Interpret which model fidelities to use for each component (if specified)
        if use_model is not None:
            if not isinstance(use_model, dict):
                use_model = {node: use_model for node in self.graph.nodes}  # use same for each component
        else:
            use_model = {node: None for node in self.graph.nodes}
        use_model = {node: use_model.get(node, None) for node in self.graph.nodes}

        # Convert system into DAG by grouping strongly-connected-components
        dag = nx.condensation(self.graph)

        # Compute component models in topological order
        for supernode in nx.topological_sort(dag):
            if np.all(list(is_computed.values())):
                break  # Exit early if all selected return qois are computed

            scc = [n for n in dag.nodes[supernode]['members']]

            # Compute single component feedforward output (no FPI needed)
            if len(scc) == 1:
                if verbose:
                    self.logger.info(f"Running component '{scc[0]}'...")
                    t1 = time.time()

                # Gather inputs
                comp = self[scc[0]]
                comp_input = {var: arr[samples.valid_idx, ...] for var, arr in all_inputs.items() if var in comp.inputs}

                # Compute outputs
                indices = index_set.get(scc[0], None) if index_set is not None else None
                if model_dir is not None:
                    output_dir = Path(model_dir) / scc[0]
                    if not output_dir.exists():
                        os.mkdir(output_dir)
                comp_output = comp.predict(comp_input, use_model=use_model.get(scc[0]), model_dir=output_dir,
                                           training=training, index_set=indices)
                for var, arr in comp_output.items():
                    output_shape = arr.shape[1:]
                    y.setdefault(var, np.full((N, *output_shape), np.nan))
                    y[var][samples.valid_idx, ...] = arr
                    samples.valid_idx = np.logical_and(samples.valid_idx, ~np.any(np.isnan(y[var]),
                                                                                  axis=tuple(range(1, y[var].ndim))))
                    is_computed[var] = True

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
                    lb, ub = var.get_domain()
                    shape = (N,) + (1,) * len(var.shape)
                    coupling_prev[var] = np.broadcast_to((lb + ub) / 2, shape).copy()

                # Main FPI loop
                if verbose:
                    self.logger.info(f"Initializing FPI for SCC {scc} ...")
                    t1 = time.time()
                k = 0
                residual_hist = deque(maxlen=anderson_mem)
                coupling_hist = deque(maxlen=anderson_mem)
                samples.reset_convergence()

                def _end_conditions_met():
                    """Compute residual, update history, and check end conditions."""
                    residual = {}
                    converged_idx = np.full(N, True)
                    for var in coupling_vars:
                        residual[var] = y[var] - coupling_prev[var]
                        var_conv = np.all(np.abs(residual[var]) <= fpi_tol, axis=tuple(range(1, residual[var].ndim)))
                        converged_idx = np.logical_and(converged_idx, var_conv)
                    samples.converged_idx = np.logical_or(samples.converged_idx, converged_idx)

                    for var in coupling_vars:
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
                        for var in coupling_vars:
                            y[var][~samples.converged_idx, ...] = np.nan
                        samples.valid_idx = np.logical_and(samples.valid_idx, samples.converged_idx)
                        return True
                    else:
                        return False

                def _merge_shapes(target_shape, arr):
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

                while True:
                    for node in scc:
                        # Gather inputs from exogenous and coupling sources
                        comp = self[node]
                        comp_input = {}
                        for var in comp.inputs:
                            if (arr := coupling_prev.get(var)) is not None:
                                comp_input[var] = arr[samples.curr_idx, ...]
                            else:
                                comp_input[var] = all_inputs[var][samples.curr_idx, ...]

                        # Compute component outputs (just don't do this FPI with the real models, please..)
                        indices = index_set.get(node, None) if index_set is not None else None
                        comp_output = comp.predict(comp_input, use_model=use_model.get(node),
                                                   model_dir=None, training=training, index_set=indices)
                        for var, arr in comp_output.items():
                            output_shape = arr.shape[1:]
                            if y.get(var) is not None:
                                if output_shape != y.get(var).shape[1:]:
                                    y[var] = _merge_shapes((N, *output_shape), y[var])
                            else:
                                y.setdefault(var, np.full((N, *output_shape), np.nan))
                            y[var][samples.curr_idx, ...] = arr

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
                    for var in coupling_vars:
                        shape = coupling_prev[var].shape[1:]
                        var_shapes.append(shape)
                        xdims.append(int(np.prod(shape)))
                    N_couple = int(np.sum(xdims))
                    res_snap = np.empty((N_curr, N_couple, mk))       # Shortened snapshot of residual history
                    coupling_snap = np.empty((N_curr, N_couple, mk))  # Shortened snapshot of coupling history
                    for i, (coupling_iter, residual_iter) in enumerate(zip(coupling_hist, residual_hist)):
                        start_idx = 0
                        for j, var in enumerate(coupling_vars):
                            end_idx = start_idx + xdims[j]
                            coupling_snap[:, start_idx:end_idx, i] = coupling_iter[var][samples.curr_idx, ...].reshape((N_curr, -1))
                            res_snap[:, start_idx:end_idx, i] = residual_iter[var][samples.curr_idx, ...].reshape((N_curr, -1))
                            start_idx = end_idx
                    C = np.ones((N_curr, 1, mk))
                    b = np.zeros((N_curr, N_couple, 1))
                    d = np.ones((N_curr, 1, 1))
                    alpha = np.expand_dims(self._constrained_lls(res_snap, b, C, d), axis=-3)   # (..., 1, mk, 1)
                    coupling_new = np.squeeze(coupling_snap[:, :, np.newaxis, :] @ alpha, axis=(-1, -2))
                    start_idx = 0
                    for j, var in enumerate(coupling_vars):
                        end_idx = start_idx + xdims[j]
                        coupling_prev[var][samples.curr_idx, ...] = coupling_new[:, start_idx:end_idx].reshape((N_curr, *var_shapes[j]))
                        start_idx = end_idx
                    k += 1

                for node in scc:
                    for var in self[node].outputs:
                        is_computed[var] = True

        # Return all component outputs; samples that didn't converge during FPI are left as np.nan
        return format_outputs(y, loop_shape)

    def __call__(self, *args, **kwargs):
        """Convenience wrapper to allow calling as `ret = SystemSurrogate(x)`."""
        return self.predict(*args, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, System):
            return False
        return (self.components == other.components and
                self.name == other.name and
                self.train_history == other.train_history)

    def __getitem__(self, component: str) -> Component:
        """Convenience method to get the `Component` object from the `SystemSurrogate`.

        :param component: the name of the component to get
        :returns: the `Component` object
        """
        return self.get_component(component)

    def get_component(self, comp_name: str) -> Component:
        """Return the `Component` object for this component.

        :param comp_name: name of the component to return
        :returns: the `Component` object
        """
        comp = self if comp_name.lower() == 'system' else self.graph.nodes[comp_name]['component']
        return comp

    def _print_title_str(self, title_str: str):
        """Log an important message."""
        self.logger.info('-' * int(len(title_str)/2) + title_str + '-' * int(len(title_str)/2))

    def _save_on_error(func):
        """Gracefully exit and save the `System` object on any errors."""
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except:
                self.save_to_file('system_error.pkl')
                self.logger.critical(f'An error occurred during execution of {func.__name__}. Saving '
                                     f'System object to system_error.pkl', exc_info=True)
                self.logger.info(f'Final system surrogate on exit: \n {self}')
                raise
        return wrap
    _save_on_error = staticmethod(_save_on_error)

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
    def load_from_file(filename: str, root_dir: str | Path = None, loader=None):
        """Load surrogate from file. Defaults to yaml loading. Tries to infer `amisc` directory structure.

        :param filename: the name of the load file
        :param root_dir: set this as the surrogate's root directory (will try to load from `amisc_` fmt by default)
        :param loader: the encoder to use (defaults to the `amisc` yaml encoder)
        """
        from amisc import YamlLoader
        encoder = loader or YamlLoader
        system = encoder.load(filename)
        root_dir = root_dir or system.root_dir

        # Try to infer amisc_root/surrogates/filename structure
        if root_dir is None:
            parts = Path(filename).resolve().parts
            if len(parts) > 2:
                if parts[-3].startswith('amisc_'):
                    root_dir = Path(filename).resolve().parent.parent
        system.root_dir = root_dir
        return system

    def serialize(self, keep_components=False, serialize_args=None, serialize_kwargs=None) -> dict:
        """Convert to a `dict` with only standard Python types for fields."""
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
                else:
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

    @staticmethod
    def _constrained_lls(A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Minimize $||Ax-b||_2$, subject to $Cx=d$, i.e. constrained linear least squares.

        !!! Note
            See http://www.seas.ucla.edu/~vandenbe/133A/lectures/cls.pdf for more detail.

        :param A: `(..., M, N)`, vandermonde matrix
        :param b: `(..., M, 1)`, data
        :param C: `(..., P, N)`, constraint operator
        :param d: `(..., P, 1)`, constraint condition
        :returns: `(..., N, 1)`, the solution parameter vector `x`
        """
        M = A.shape[-2]
        dims = len(A.shape[:-2])
        T_axes = tuple(np.arange(0, dims)) + (-1, -2)
        Q, R = np.linalg.qr(np.concatenate((A, C), axis=-2))
        Q1 = Q[..., :M, :]
        Q2 = Q[..., M:, :]
        Q1_T = np.transpose(Q1, axes=T_axes)
        Q2_T = np.transpose(Q2, axes=T_axes)
        Qtilde, Rtilde = np.linalg.qr(Q2_T)
        Qtilde_T = np.transpose(Qtilde, axes=T_axes)
        Rtilde_T_inv = np.linalg.pinv(np.transpose(Rtilde, axes=T_axes))
        w = np.linalg.pinv(Rtilde) @ (Qtilde_T @ Q1_T @ b - Rtilde_T_inv @ d)

        return np.linalg.pinv(R) @ (Q1_T @ b - Q2_T @ w)


# class SystemSurrogate:
#     """Multidisciplinary (MD) surrogate framework top-level class.
#
#     !!! Note "Accessing individual components"
#         The `Component` objects that compose `SystemSurrogate` are internally stored in the `self.graph.nodes`
#         data structure. You can access them with `get_component(comp_name)`.
#
#     :ivar exo_vars: global list of exogenous/external inputs for the MD system
#     :ivar coupling_vars: global list of coupling variables for the MD system (including all system-level outputs)
#     :ivar refine_level: the total number of refinement steps that have been made
#     :ivar build_metrics: contains data that summarizes surrogate training progress
#     :ivar root_dir: root directory where all surrogate build products are saved to file
#     :ivar log_file: log file where all logs are written to by default
#     :ivar executor: manages parallel execution for the system
#     :ivar graph: the internal graph data structure of the MD system
#
#     :vartype exo_vars: list[Variable]
#     :vartype coupling_vars: list[Variable]
#     :vartype refine_level: int
#     :vartype build_metrics: dict
#     :vartype root_dir: str
#     :vartype log_file: str
#     :vartype executor: Executor
#     :vartype graph: nx.DiGraph
#     """
#
#     def __init__(self, components: list[Component] | Component, exo_vars: list[Variable] | Variable,
#                  coupling_vars: list[Variable] | Variable, est_bds: int = 0, save_dir: str | Path = None,
#                  executor: Executor = None, stdout: bool = True, init_surr: bool = True, logger_name: str = None):
#         # Estimate coupling variable bounds
#         if est_bds > 0:
#             self._estimate_coupling_bds(est_bds)
#
#         # Init system with most coarse fidelity indices in each component
#         if init_surr:
#             self.init_system()
#         self._save_progress('sys_init.pkl')
#
#     def init_system(self):
#         """Add the coarsest multi-index to each component surrogate."""
#         self._print_title_str('Initializing all component surrogates')
#         for node, node_obj in self.graph.nodes.items():
#             node_obj['surrogate'].init_coarse()
#             # for alpha, beta in list(node_obj['surrogate'].candidate_set):
#             #     # Add one refinement in each input dimension to initialize
#             #     node_obj['surrogate'].activate_index(alpha, beta)
#             self.logger.info(f"Initialized component '{node}'.")
#
#     def fit(self, qoi_ind = None, num_refine: int = 100, max_iter: int = 20, max_tol: float = 1e-3,
#             max_runtime: float = 1, save_interval: int = 0, update_bounds: bool = True, test_set: dict = None,
#             n_jobs: int = 1):
#         """Train the system surrogate adaptively by iterative refinement until an end condition is met.
#
#         :param qoi_ind: list of system QoI variables to focus refinement on, use all QoI if not specified
#         :param num_refine: number of samples of exogenous inputs to compute error indicators on
#         :param max_iter: the maximum number of refinement steps to take
#         :param max_tol: the max allowable value in relative L2 error to achieve
#         :param max_runtime: the maximum wall clock time (hr) to run refinement for (will go until all models finish)
#         :param save_interval: number of refinement steps between each progress save, none if 0
#         :param update_bounds: whether to continuously update coupling variable bounds during refinement
#         :param test_set: `dict(xt=(Nt, x_dim), yt=(Nt, y_dim)` to show convergence of surrogate to the truth model
#         :param n_jobs: number of cpu workers for computing error indicators (on master MPI task), 1=sequential
#         """
#         qoi_ind = self._get_qoi_ind(qoi_ind)
#         Nqoi = len(qoi_ind)
#         max_iter = self.refine_level + max_iter
#         curr_error = np.inf
#         t_start = time.time()
#         test_stats, xt, yt, t_fig, t_ax = None, None, None, None, None
#
#         # Record of (error indicator, component, alpha, beta, num_evals, total added cost (s)) for each iteration
#         train_record = self.build_metrics.get('train_record', [])
#         if test_set is not None:
#             xt, yt = test_set['xt'], test_set['yt']
#         xt, yt = self.build_metrics.get('xt', xt), self.build_metrics.get('yt', yt)  # Overrides test set param
#
#         # Track convergence progress on a test set and on the max error indicator
#         err_fig, err_ax = plt.subplots()
#         if xt is not None and yt is not None:
#             self.build_metrics['xt'] = xt
#             self.build_metrics['yt'] = yt
#             if self.build_metrics.get('test_stats') is not None:
#                 test_stats = self.build_metrics.get('test_stats')
#             else:
#                 # Get initial perf metrics, (2, Nqoi)
#                 test_stats = np.expand_dims(self.get_test_metrics(xt, yt, qoi_ind=qoi_ind), axis=0)
#             t_fig, t_ax = plt.subplots(1, Nqoi) if Nqoi > 1 else plt.subplots()
#
#         # Set up a parallel pool of workers, sequential if n_jobs=1
#         with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
#             while True:
#                 # Check all end conditions
#                 if self.refine_level >= max_iter:
#                     self._print_title_str(f'Termination criteria reached: Max iteration {self.refine_level}/{max_iter}')
#                     break
#                 if curr_error == -np.inf:
#                     self._print_title_str('Termination criteria reached: No candidates left to refine')
#                     break
#                 if curr_error < max_tol:
#                     self._print_title_str(f'Termination criteria reached: L2 error {curr_error} < tol {max_tol}')
#                     break
#                 if ((time.time() - t_start)/3600.0) >= max_runtime:
#                     actual = datetime.timedelta(seconds=time.time()-t_start)
#                     target = datetime.timedelta(seconds=max_runtime*3600)
#                     self._print_title_str(f'Termination criteria reached: runtime {str(actual)} > {str(target)}')
#                     break
#
#                 # Refine surrogate and save progress
#                 refine_res = self.refine(qoi_ind=qoi_ind, num_refine=num_refine, update_bounds=update_bounds,
#                                          ppool=ppool)
#                 curr_error = refine_res[0]
#                 if save_interval > 0 and self.refine_level % save_interval == 0:
#                     self._save_progress(f'sys_iter_{self.refine_level}.pkl')
#
#                 # Plot progress of error indicator
#                 train_record.append(refine_res)
#                 error_record = [res[0] for res in train_record]
#                 self.build_metrics['train_record'] = train_record
#                 err_ax.clear(); err_ax.grid(); err_ax.plot(error_record, '-k')
#                 ax_default(err_ax, 'Iteration', r'Relative $L_2$ error indicator', legend=False)
#                 err_ax.set_yscale('log')
#                 if self.root_dir is not None:
#                     err_fig.savefig(str(Path(self.root_dir) / 'error_indicator.png'), dpi=300, format='png')
#
#                 # Plot progress on test set
#                 if xt is not None and yt is not None:
#                     stats = self.get_test_metrics(xt, yt, qoi_ind=qoi_ind)
#                     test_stats = np.concatenate((test_stats, stats[np.newaxis, ...]), axis=0)
#                     for i in range(Nqoi):
#                         ax = t_ax if Nqoi == 1 else t_ax[i]
#                         ax.clear(); ax.grid(); ax.set_yscale('log')
#                         ax.plot(test_stats[:, 1, i], '-k')
#                         ax.set_title(self.coupling_vars[qoi_ind[i]].get_tex(units=True))
#                         ax_default(ax, 'Iteration', r'Relative $L_2$ error', legend=False)
#                     t_fig.set_size_inches(3.5*Nqoi, 3.5)
#                     t_fig.tight_layout()
#                     if self.root_dir is not None:
#                         t_fig.savefig(str(Path(self.root_dir) / 'test_set.png'), dpi=300, format='png')
#                     self.build_metrics['test_stats'] = test_stats
#
#         self._save_progress('sys_final.pkl')
#         self.logger.info(f'Final system surrogate: \n {self}')
#
#     def get_allocation(self, idx: int = None):
#         """Get a breakdown of cost allocation up to a certain iteration number during training (starting at 1).
#
#         :param idx: the iteration number to get allocation results for (defaults to last refinement step)
#         :returns: `cost_alloc, offline_alloc, cost_cum` - the cost alloc per node/fidelity and cumulative training cost
#         """
#         if idx is None:
#             idx = self.refine_level
#         if idx > self.refine_level:
#             raise ValueError(f'Specified index: {idx} is greater than the max training level of {self.refine_level}')
#
#         cost_alloc = dict()     # Cost allocation per node and model fidelity
#         cost_cum = [0.0]        # Cumulative cost allocation during training
#
#         # Add initialization costs for each node
#         for node, node_obj in self.graph.nodes.items():
#             surr = node_obj['surrogate']
#             base_alpha = (0,) * len(surr.truth_alpha)
#             base_beta = (0,) * (len(surr.max_refine) - len(surr.truth_alpha))
#             base_cost = surr.get_cost(base_alpha, base_beta)
#             cost_alloc[node] = dict()
#             if base_cost > 0:
#                 cost_alloc[node][str(base_alpha)] = np.array([1, float(base_cost)])
#                 cost_cum[0] += float(base_cost)
#
#         # Add cumulative training costs
#         for i in range(idx):
#             err_indicator, node, alpha, beta, num_evals, cost = self.build_metrics['train_record'][i]
#             if cost_alloc[node].get(str(alpha), None) is None:
#                 cost_alloc[node][str(alpha)] = np.zeros(2)  # (num model evals, total cpu_time cost)
#             cost_alloc[node][str(alpha)] += [round(num_evals), float(cost)]
#             cost_cum.append(float(cost))
#
#         # Get summary of total offline costs spent building search candidates (i.e. training overhead)
#         offline_alloc = dict()
#         for node, node_obj in self.graph.nodes.items():
#             surr = node_obj['surrogate']
#             offline_alloc[node] = dict()
#             for alpha, beta in surr.candidate_set:
#                 if offline_alloc[node].get(str(alpha), None) is None:
#                     offline_alloc[node][str(alpha)] = np.zeros(2)   # (num model evals, total cpu_time cost)
#                 added_cost = surr.get_cost(alpha, beta)
#                 base_cost = surr.get_sub_surrogate(alpha, beta).model_cost
#                 offline_alloc[node][str(alpha)] += [round(added_cost/base_cost), float(added_cost)]
#
#         return cost_alloc, offline_alloc, np.cumsum(cost_cum)
#
#     def get_test_metrics(self, xt: np.ndarray, yt: np.ndarray, qoi_ind = None,
#                          training: bool = True) -> np.ndarray:
#         """Get relative L2 error metric over a test set.
#
#         :param xt: `(Nt, x_dim)` random test set of inputs
#         :param yt: `(Nt, y_dim)` random test set outputs
#         :param qoi_ind: list of indices of QoIs to get metrics for
#         :param training: whether to evaluate the surrogate in training or evaluation mode
#         :returns: `stats` - `(2, Nqoi)` array &rarr; `[num_candidates, rel_L2_error]` for each QoI
#         """
#         qoi_ind = self._get_qoi_ind(qoi_ind)
#         ysurr = self(xt, training=training)
#         ysurr = ysurr[:, qoi_ind]
#         yt = yt[:, qoi_ind]
#         with np.errstate(divide='ignore', invalid='ignore'):
#             rel_l2_err = np.sqrt(np.mean((yt - ysurr) ** 2, axis=0)) / np.sqrt(np.mean(yt ** 2, axis=0))
#             rel_l2_err = np.nan_to_num(rel_l2_err, posinf=np.nan, neginf=np.nan, nan=np.nan)
#         num_cands = 0
#         for node, node_obj in self.graph.nodes.items():
#             num_cands += len(node_obj['surrogate'].index_set) + len(node_obj['surrogate'].candidate_set)
#
#         # Get test stats for each QoI
#         stats = np.zeros((2, yt.shape[-1]))
#         self.logger.debug(f'{"QoI idx": >10} {"Iteration": >10} {"len(I_k)": >10} {"Relative L2": >15}')
#         for i in range(yt.shape[-1]):
#             stats[:, i] = np.array([num_cands, rel_l2_err[i]])
#             self.logger.debug(f'{i: 10d} {self.refine_level: 10d} {num_cands: 10d} {rel_l2_err[i]: 15.5f}')
#
#         return stats
#
#     def refine(self, qoi_ind = None, num_refine: int = 100, update_bounds: bool = True,
#                ppool: Parallel = None) -> tuple:
#         """Find and refine the component surrogate with the largest error on system-level QoI.
#
#         :param qoi_ind: indices of system QoI to focus surrogate refinement on, use all QoI if not specified
#         :param num_refine: number of samples of exogenous inputs to compute error indicators on
#         :param update_bounds: whether to continuously update coupling variable bounds
#         :param ppool: a `Parallel` instance from `joblib` to compute error indicators in parallel, None=sequential
#         :returns refine_res: a tuple of `(error_indicator, component, node_star, alpha_star, beta_star, N, cost)`
#                              indicating the chosen candidate index and incurred cost
#         """
#         self._print_title_str(f'Refining system surrogate: iteration {self.refine_level + 1}')
#         set_loky_pickler('dill')    # Dill can serialize 'self' for parallel workers
#         temp_exc = self.executor    # It can't serialize an executor though, so must save this temporarily
#         self.set_executor(None)
#         qoi_ind = self._get_qoi_ind(qoi_ind)
#
#         # Compute entire integrated-surrogate on a random test set for global system QoI error estimation
#         x_exo = self.sample_inputs((num_refine,))
#         y_curr = self(x_exo, training=True)
#         y_min, y_max = None, None
#         if update_bounds:
#             y_min = np.min(y_curr, axis=0, keepdims=True)  # (1, ydim)
#             y_max = np.max(y_curr, axis=0, keepdims=True)  # (1, ydim)
#
#         # Find the candidate surrogate with the largest error indicator
#         error_max, error_indicator = -np.inf, -np.inf
#         node_star, alpha_star, beta_star, l2_star, cost_star = None, None, None, -np.inf, 0
#         for node, node_obj in self.graph.nodes.items():
#             self.logger.info(f"Estimating error for component '{node}'...")
#             candidates = node_obj['surrogate'].candidate_set.copy()
#
#             def compute_error(alpha, beta):
#                 # Helper function for computing error indicators for a given candidate (alpha, beta)
#                 index_set = node_obj['surrogate'].index_set.copy()
#                 index_set.append((alpha, beta))
#                 y_cand = self(x_exo, training=True, index_set={node: index_set})
#                 ymin = np.min(y_cand, axis=0, keepdims=True)
#                 ymax = np.max(y_cand, axis=0, keepdims=True)
#                 error = y_cand[:, qoi_ind] - y_curr[:, qoi_ind]
#                 rel_l2 = np.sqrt(np.nanmean(error ** 2, axis=0)) / np.sqrt(np.nanmean(y_cand[:, qoi_ind] ** 2, axis=0))
#                 rel_l2 = np.nan_to_num(rel_l2, nan=np.nan, posinf=np.nan, neginf=np.nan)
#                 delta_error = np.nanmax(rel_l2)  # Max relative L2 error over all system QoIs
#                 delta_work = max(1, node_obj['surrogate'].get_cost(alpha, beta))  # Cpu time (s)
#
#                 return ymin, ymax, delta_error, delta_work
#
#             if len(candidates) > 0:
#                 ret = ppool(delayed(compute_error)(alpha, beta) for alpha, beta in candidates) if ppool is not None \
#                     else [compute_error(alpha, beta) for alpha, beta in candidates]
#
#                 for i, (ymin, ymax, d_error, d_work) in enumerate(ret):
#                     if update_bounds:
#                         y_min = np.min(np.concatenate((y_min, ymin), axis=0), axis=0, keepdims=True)
#                         y_max = np.max(np.concatenate((y_max, ymax), axis=0), axis=0, keepdims=True)
#                     alpha, beta = candidates[i]
#                     error_indicator = d_error / d_work
#                     self.logger.info(f"Candidate multi-index: {(alpha, beta)}. L2 error: {d_error}. Error indicator: "
#                                      f"{error_indicator}.")
#
#                     if error_indicator > error_max:
#                         error_max = error_indicator
#                         node_star, alpha_star, beta_star, l2_star, cost_star = node, alpha, beta, d_error, d_work
#             else:
#                 self.logger.info(f"Component '{node}' has no available candidates left!")
#
#         # Update all coupling variable ranges
#         if update_bounds:
#             for i in range(y_curr.shape[-1]):
#                 self._update_coupling_bds(i, (y_min[0, i], y_max[0, i]))
#
#         # Add the chosen multi-index to the chosen component
#         self.set_executor(temp_exc)
#         if node_star is not None:
#             self.logger.info(f"Candidate multi-index {(alpha_star, beta_star)} chosen for component '{node_star}'")
#             self.graph.nodes[node_star]['surrogate'].activate_index(alpha_star, beta_star)
#             self.refine_level += 1
#             num_evals = round(cost_star / self[node_star].get_sub_surrogate(alpha_star, beta_star).model_cost)
#         else:
#             self.logger.info(f"No candidates left for refinement, iteration: {self.refine_level}")
#             num_evals = 0
#
#         return l2_star, node_star, alpha_star, beta_star, num_evals, cost_star
#
#     def _estimate_coupling_bds(self, num_est: int, anderson_mem: int = 10, fpi_tol: float = 1e-10,
#                                max_fpi_iter: int = 100):
#         """Estimate and set the coupling variable bounds.
#
#         :param num_est: the number of samples of exogenous inputs to use
#         :param anderson_mem: FPI hyperparameter (default is usually good)
#         :param fpi_tol: floating point tolerance for FPI convergence
#         :param max_fpi_iter: maximum number of FPI iterations
#         """
#         self._print_title_str('Estimating coupling variable bounds')
#         x = self.sample_inputs((num_est,))
#         y = self(x, use_model='best', verbose=True, anderson_mem=anderson_mem, fpi_tol=fpi_tol,
#                  max_fpi_iter=max_fpi_iter)
#         for i in range(len(self.coupling_vars)):
#             lb = np.nanmin(y[:, i])
#             ub = np.nanmax(y[:, i])
#             self._update_coupling_bds(i, (lb, ub), init=True)
#
#     def _update_coupling_bds(self, global_idx: int, bds: tuple, init: bool = False, buffer: float = 0.05):
#         """Update coupling variable bounds.
#
#         :param global_idx: global index of coupling variable to update
#         :param bds: new bounds to update the current bounds with
#         :param init: whether to set new bounds or update existing (default)
#         :param buffer: fraction of domain length to buffer upper/lower bounds
#         """
#         offset = buffer * (bds[1] - bds[0])
#         offset_bds = (bds[0] - offset, bds[1] + offset)
#         coupling_bds = [rv.get_domain() for rv in self.coupling_vars]
#         new_bds = offset_bds if init else (min(coupling_bds[global_idx][0], offset_bds[0]),
#                                            max(coupling_bds[global_idx][1], offset_bds[1]))
#         self.coupling_vars[global_idx].update(domain=new_bds)
#
#         # Iterate over all components and update internal coupling variable bounds
#         for node_name, node_obj in self.graph.nodes.items():
#             if global_idx in node_obj['global_in']:
#                 # Get the local index for this coupling variable within each component's inputs
#                 local_idx = len(node_obj['exo_in']) + node_obj['global_in'].index(global_idx)
#                 node_obj['surrogate'].update_input_bds(local_idx, new_bds)
#
#     def plot_slice(self, slice_idx = None, qoi_idx = None, show_surr: bool = True,
#                    show_model: list = None, model_dir: str | Path = None, N: int = 50, nominal: dict[str: float] = None,
#                    random_walk: bool = False, from_file: str | Path = None):
#         """Helper function to plot 1d slices of the surrogate and/or model(s) over the inputs.
#
#         :param slice_idx: list of exogenous input variables or indices to take 1d slices of
#         :param qoi_idx: list of model output variables or indices to plot 1d slices of
#         :param show_surr: whether to show the surrogate prediction
#         :param show_model: also plot model predictions, list() of ['best', 'worst', tuple(alpha), etc.]
#         :param model_dir: base directory to save model outputs (if specified)
#         :param N: the number of points to take in the 1d slice
#         :param nominal: `dict` of `str(var)->nominal` to use as constant value for all non-sliced variables
#         :param random_walk: whether to slice in a random d-dimensional direction or hold all params const while slicing
#         :param from_file: path to a .pkl file to load a saved slice from disk
#         :returns: `fig, ax` with `num_slice` by `num_qoi` subplots
#         """
#         # Manage loading important quantities from file (if provided)
#         xs, ys_model, ys_surr = None, None, None
#         if from_file is not None:
#             with open(Path(from_file), 'rb') as fd:
#                 slice_data = pickle.load(fd)
#                 slice_idx = slice_data['slice_idx']     # Must use same input slices as save file
#                 show_model = slice_data['show_model']   # Must use same model data as save file
#                 qoi_idx = slice_data['qoi_idx'] if qoi_idx is None else qoi_idx
#                 xs = slice_data['xs']
#                 model_dir = None  # Don't run or save any models if loading from file
#
#         # Set default values (take up to the first 3 slices by default)
#         rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
#         if model_dir is not None:
#             os.mkdir(Path(model_dir) / f'sweep_{rand_id}')
#         if nominal is None:
#             nominal = dict()
#         slice_idx = list(np.arange(0, min(3, len(self.exo_vars)))) if slice_idx is None else slice_idx
#         qoi_idx = list(np.arange(0, min(3, len(self.coupling_vars)))) if qoi_idx is None else qoi_idx
#         if isinstance(slice_idx[0], str | Variable):
#             slice_idx = [self.exo_vars.index(var) for var in slice_idx]
#         if isinstance(qoi_idx[0], str | Variable):
#             qoi_idx = [self.coupling_vars.index(var) for var in qoi_idx]
#
#         exo_bds = [var.get_domain() for var in self.exo_vars]
#         xlabels = [self.exo_vars[idx].get_tex(units=True) for idx in slice_idx]
#         ylabels = [self.coupling_vars[idx].get_tex(units=True) for idx in qoi_idx]
#
#         # Construct slice model inputs (if not provided)
#         if xs is None:
#             xs = np.zeros((N, len(slice_idx), len(self.exo_vars)))
#             for i in range(len(slice_idx)):
#                 if random_walk:
#                     # Make a random straight-line walk across d-cube
#                     r0 = np.squeeze(self.sample_inputs((1,), use_pdf=False), axis=0)
#                     r0[slice_idx[i]] = exo_bds[slice_idx[i]][0]             # Start slice at this lower bound
#                     rf = np.squeeze(self.sample_inputs((1,), use_pdf=False), axis=0)
#                     rf[slice_idx[i]] = exo_bds[slice_idx[i]][1]             # Slice up to this upper bound
#                     xs[0, i, :] = r0
#                     for k in range(1, N):
#                         xs[k, i, :] = xs[k-1, i, :] + (rf-r0)/(N-1)
#                 else:
#                     # Otherwise, only slice one variable
#                     for j in range(len(self.exo_vars)):
#                         if j == slice_idx[i]:
#                             xs[:, i, j] = np.linspace(exo_bds[slice_idx[i]][0], exo_bds[slice_idx[i]][1], N)
#                         else:
#                             xs[:, i, j] = nominal.get(self.exo_vars[j], self.exo_vars[j].nominal)
#
#         # Walk through each model that is requested by show_model
#         if show_model is not None:
#             if from_file is not None:
#                 ys_model = slice_data['ys_model']
#             else:
#                 ys_model = list()
#                 for model in show_model:
#                     output_dir = None
#                     if model_dir is not None:
#                         output_dir = (Path(model_dir) / f'sweep_{rand_id}' /
#                                       str(model).replace('{', '').replace('}', '').replace(':', '=').replace("'", ''))
#                         os.mkdir(output_dir)
#                     ys_model.append(self(xs, use_model=model, model_dir=output_dir))
#         if show_surr:
#             ys_surr = self(xs) if from_file is None else slice_data['ys_surr']
#
#         # Make len(qoi) by len(inputs) grid of subplots
#         fig, axs = plt.subplots(len(qoi_idx), len(slice_idx), sharex='col', sharey='row')
#         for i in range(len(qoi_idx)):
#             for j in range(len(slice_idx)):
#                 if len(qoi_idx) == 1:
#                     ax = axs if len(slice_idx) == 1 else axs[j]
#                 elif len(slice_idx) == 1:
#                     ax = axs if len(qoi_idx) == 1 else axs[i]
#                 else:
#                     ax = axs[i, j]
#                 x = xs[:, j, slice_idx[j]]
#                 if show_model is not None:
#                     c = np.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1]]) if len(show_model) <= 2 else (
#                         plt.get_cmap('jet')(np.linspace(0, 1, len(show_model))))
#                     for k in range(len(show_model)):
#                         model_str = (str(show_model[k]).replace('{', '').replace('}', '')
#                                      .replace(':', '=').replace("'", ''))
#                         model_ret = ys_model[k]
#                         y_model = model_ret[:, j, qoi_idx[i]]
#                         label = {'best': 'High-fidelity' if len(show_model) > 1 else 'Model',
#                                  'worst': 'Low-fidelity'}.get(model_str, model_str)
#                         ax.plot(x, y_model, ls='-', c=c[k, :], label=label)
#                 if show_surr:
#                     y_surr = ys_surr[:, j, qoi_idx[i]]
#                     ax.plot(x, y_surr, '--r', label='Surrogate')
#                 ylabel = ylabels[i] if j == 0 else ''
#                 xlabel = xlabels[j] if i == len(qoi_idx) - 1 else ''
#                 legend = (i == 0 and j == len(slice_idx) - 1)
#                 ax_default(ax, xlabel, ylabel, legend=legend)
#         fig.set_size_inches(3 * len(slice_idx), 3 * len(qoi_idx))
#         fig.tight_layout()
#
#         # Save results (unless we were already loading from a save file)
#         if from_file is None and self.root_dir is not None:
#             fname = f's{",".join([str(i) for i in slice_idx])}_q{",".join([str(i) for i in qoi_idx])}'
#             fname = f'sweep_rand{rand_id}_' + fname if random_walk else f'sweep_nom{rand_id}_' + fname
#             fdir = Path(self.root_dir) if model_dir is None else Path(model_dir) / f'sweep_{rand_id}'
#             fig.savefig(fdir / f'{fname}.png', dpi=300, format='png')
#             save_dict = {'slice_idx': slice_idx, 'qoi_idx': qoi_idx, 'show_model': show_model, 'show_surr': show_surr,
#                          'nominal': nominal, 'random_walk': random_walk, 'xs': xs, 'ys_model': ys_model,
#                          'ys_surr': ys_surr}
#             with open(fdir / f'{fname}.pkl', 'wb') as fd:
#                 pickle.dump(save_dict, fd)
#
#         return fig, axs
#
#     def plot_allocation(self, cmap: str = 'Blues', text_bar_width: float = 0.06, arrow_bar_width: float = 0.02):
#         """Plot bar charts showing cost allocation during training.
#
#         !!! Warning "Beta feature"
#             This has pretty good default settings, but it might look terrible for your use. Mostly provided here as
#             a template for making cost allocation bar charts. Please feel free to copy and edit in your own code.
#
#         :param cmap: the colormap string identifier for `plt`
#         :param text_bar_width: the minimum total cost fraction above which a bar will print centered model fidelity text
#         :param arrow_bar_width: the minimum total cost fraction above which a bar will try to print text with an arrow;
#                                 below this amount, the bar is too skinny and won't print any text
#         :returns: `fig, ax`, Figure and Axes objects
#         """
#         # Get total cost (including offline overhead)
#         train_alloc, offline_alloc, cost_cum = self.get_allocation()
#         total_cost = cost_cum[-1]
#         for node, alpha_dict in offline_alloc.items():
#             for alpha, cost in alpha_dict.items():
#                 total_cost += cost[1]
#
#         # Remove nodes with cost=0 from alloc dicts (i.e. analytical models)
#         remove_nodes = []
#         for node, alpha_dict in train_alloc.items():
#             if len(alpha_dict) == 0:
#                 remove_nodes.append(node)
#         for node in remove_nodes:
#             del train_alloc[node]
#             del offline_alloc[node]
#
#         # Bar chart showing cost allocation breakdown for MF system at end
#         fig, axs = plt.subplots(1, 2, sharey='row')
#         width = 0.7
#         x = np.arange(len(train_alloc))
#         xlabels = list(train_alloc.keys())
#         cmap = plt.get_cmap(cmap)
#         for k in range(2):
#             ax = axs[k]
#             alloc = train_alloc if k == 0 else offline_alloc
#             ax.set_title('Online training' if k == 0 else 'Overhead')
#             for j, (node, alpha_dict) in enumerate(alloc.items()):
#                 bottom = 0
#                 c_intervals = np.linspace(0, 1, len(alpha_dict))
#                 bars = [(alpha, cost, cost[1] / total_cost) for alpha, cost in alpha_dict.items()]
#                 bars = sorted(bars, key=lambda ele: ele[2], reverse=True)
#                 for i, (alpha, cost, frac) in enumerate(bars):
#                     p = ax.bar(x[j], frac, width, color=cmap(c_intervals[i]), linewidth=1,
#                                edgecolor=[0, 0, 0], bottom=bottom)
#                     bottom += frac
#                     if frac > text_bar_width:
#                         ax.bar_label(p, labels=[f'{alpha}, {round(cost[0])}'], label_type='center')
#                     elif frac > arrow_bar_width:
#                         xy = (x[j] + width / 2, bottom - frac / 2)  # Label smaller bars with a text off to the side
#                         ax.annotate(f'{alpha}, {round(cost[0])}', xy, xytext=(xy[0] + 0.2, xy[1]),
#                                     arrowprops={'arrowstyle': '->', 'linewidth': 1})
#                     else:
#                         pass  # Don't label really small bars
#             ax_default(ax, '', "Fraction of total cost" if k == 0 else '', legend=False)
#             ax.set_xticks(x, xlabels)
#             ax.set_xlim(left=-1, right=x[-1] + 1)
#         fig.set_size_inches(8, 4)
#         fig.tight_layout()
#
#         if self.root_dir is not None:
#             fig.savefig(Path(self.root_dir) / 'mf_allocation.png', dpi=300, format='png')
#
#         return fig, axs
