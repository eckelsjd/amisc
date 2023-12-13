"""`system.py`

The `SystemSurrogate` is a framework for multidisciplinary models. It manages multiple single discipline component
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
    The return value is a Python dictionary of the form `ret = {'y': y, 'files': files, 'cost': cost, etc.}`.
    In the return dictionary, you specify the raw model output `y` as an `np.ndarray` at a _minimum_. Optionally, you can
    specify paths to output files and the average model cost (in seconds of cpu time), and anything else you want. Your
    `model()` function can do anything it wants in order to go from `x` &rarr; `y`. Python has the flexibility to call
    virtually any external codes, or to implement the function natively with `numpy`.

!!! Info "Component specification"
    A component adds some extra configuration around a callable `model`. These configurations are defined in a Python
    dictionary, which we give the custom type `ComponentSpec`. At a bare _minimum_, you must specify a callable
    `model` and its connections to other models within the multidisciplinary system. The limiting case is a single
    component model, for which the configuration is simply `component = ComponentSpec(model)`.
"""
import os
import time
import datetime
import functools
import copy
from datetime import timezone
from pathlib import Path
import random
import string
import pickle
from collections import UserDict
from concurrent.futures import Executor

import numpy as np
import networkx as nx
import dill
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

from amisc.component import SparseGridSurrogate, ComponentSurrogate, AnalyticalSurrogate
from amisc import IndicesRV, IndexSet
from amisc.utils import ax_default, get_logger
from amisc.rv import BaseRV


class ComponentSpec(UserDict):
    """Provides a simple extension class of a Python dictionary, used to configure a component model.

    !!! Info "Specifying a list of random variables"
        The three fields: `exo_in`, `coupling_in`, and `coupling_out` fully determine how a component fits within a
        multidisciplinary system. For each, you must specify a list of variables in the same order as the model uses
        them. The model will use all exogenous inputs first, and then all coupling inputs. You can use a variable's
        global integer index into the system `exo_vars` or `coupling_vars`, or you can use the `str` id of the variable
        or the variable itself. This is summarized in the `amisc.IndicesRV` custom type.

    !!! Example
        Let's say you have a model:
        ```python
        def my_model(x, *args, **kwargs):
            print(x.shape)  # (3,), so a total of 3 inputs
            G = 6.674e-11
            m1 = x[0]           # System-level input
            m2 = x[1]           # System-level input
            r = x[2]            # Coupling input
            F = G*m1*m2 / r**2
            return {'y': F}
        ```
        Let's say this model is part of a larger system where `m1` and `m2` are specified by the system, and `r` comes
        from a different model that predicts the distance between two objects. You would set the configuration as:
        ```python
        component = ComponentSpec(my_model, exo_in=['m1', 'm2'], coupling_in=['r'], coupling_out=['F'])
        ```
    """
    Options = ['model', 'name', 'exo_in', 'coupling_in', 'coupling_out', 'truth_alpha', 'max_alpha', 'max_beta',
               'surrogate', 'model_args', 'model_kwargs', 'save_output']

    def __init__(self, model: callable, name: str = '', exo_in: IndicesRV = None,
                 coupling_in: IndicesRV | dict[str: IndicesRV] = None, coupling_out: IndicesRV = None,
                 truth_alpha: tuple | int = (), max_alpha: tuple | int = (), max_beta: tuple | int = (),
                 surrogate: str | ComponentSurrogate = 'lagrange', model_args: tuple = (), model_kwargs: dict = None,
                 save_output: bool = False):
        """Construct the configuration for this component model.

        !!! Warning
            Always specify the model at a _global_ scope, i.e. don't use `lambda` or nested functions. When saving to
            file, only a symbolic reference to the function signature will be saved, which must be globally defined
            when loading back from that save file.

        :param model: the component model, must be defined in a global scope (i.e. in a module or top-level of a script)
        :param name: the name used to identify this component model
        :param exo_in: specifies the global, system-level (i.e. exogenous/external) inputs to this model
        :param coupling_in: specifies the coupling inputs received from other models
        :param coupling_out: specifies all outputs of this model (which may couple later to downstream models)
        :param truth_alpha: the model fidelity indices to treat as a "ground truth" reference
        :param max_alpha: the maximum model fidelity indices to allow for refinement purposes
        :param max_beta: the maximum surrogate fidelity indices to allow for refinement purposes
        :param surrogate: one of ('lagrange, 'analytical'), or the `ComponentSurrogate` class to use directly
        :param model_args: optional arguments to pass to the component model
        :param model_kwargs: optional keyword arguments to pass to the component model
        :param save_output: whether this model will be saving outputs to file
        """
        d = locals()
        d2 = {key: value for key, value in d.items() if key in ComponentSpec.Options}
        super().__init__(d2)

    def __setitem__(self, key, value):
        if key in ComponentSpec.Options:
            super().__setitem__(key, value)
        else:
            raise ValueError(f'"{key}" is not applicable for a ComponentSpec. Try one of {ComponentSpec.Options}.')

    def __delitem__(self, key):
        raise TypeError("Not allowed to delete items from a ComponentSpec.")


class SystemSurrogate:
    """Multidisciplinary (MD) surrogate framework top-level class.

    !!! Note "Accessing individual components"
        The `ComponentSurrogate` objects that compose `SystemSurrogate` are internally stored in the `self.graph.nodes`
        data structure. You can access them with `get_component(comp_name)`.

    :ivar exo_vars: global list of exogenous/external inputs for the MD system
    :ivar coupling_vars: global list of coupling variables for the MD system (including all system-level outputs)
    :ivar refine_level: the total number of refinement steps that have been made
    :ivar build_metrics: contains data that summarizes surrogate training progress
    :ivar root_dir: root directory where all surrogate build products are saved to file
    :ivar log_file: log file where all logs are written to by default
    :ivar executor: manages parallel execution for the system
    :ivar graph: the internal graph data structure of the MD system

    :vartype exo_vars: list[BaseRV]
    :vartype coupling_vars: list[BaseRV]
    :vartype refine_level: int
    :vartype build_metrics: dict
    :vartype root_dir: str
    :vartype log_file: str
    :vartype executor: Executor
    :vartype graph: nx.DiGraph
    """

    def __init__(self, components: list[ComponentSpec] | ComponentSpec, exo_vars: list[BaseRV] | BaseRV,
                 coupling_vars: list[BaseRV] | BaseRV, est_bds: int = 0, save_dir: str | Path = None,
                 executor: Executor = None, stdout: bool = True, init_surr: bool = True):
        """Construct the MD system surrogate.

        !!! Warning
            Component models should always use coupling variables in the order they appear in the system-level
            `coupling_vars`.

        :param components: list of components in the MD system (using the ComponentSpec class)
        :param exo_vars: list of system-level exogenous/external inputs
        :param coupling_vars: list of all coupling variables (including all system-level outputs)
        :param est_bds: number of samples to estimate coupling variable bounds, do nothing if 0
        :param save_dir: root directory for all build products (.log, .pkl, .json, etc.), won't save if None
        :param executor: an instance of a `concurrent.futures.Executor`, use to iterate new candidates in parallel
        :param stdout: whether to log to console
        :param init_surr: whether to initialize the surrogate immediately when constructing
        """
        # Setup root save directory
        if save_dir is not None:
            timestamp = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.')
            save_dir = Path(save_dir) / ('amisc_' + timestamp)
            os.mkdir(save_dir)
            self.root_dir = str(save_dir.resolve())
            os.mkdir(Path(self.root_dir) / 'sys')
            os.mkdir(Path(self.root_dir) / 'components')
            fname = timestamp + 'UTC_sys.log'
            self.log_file = str((Path(self.root_dir) / fname).resolve())
        else:
            self.root_dir = None
            self.log_file = None
        self.logger = get_logger(self.__class__.__name__, log_file=self.log_file, stdout=stdout)
        self.executor = executor

        # Store system info in a directed graph data structure
        self.graph = nx.DiGraph()
        self.exo_vars = copy.deepcopy(exo_vars) if isinstance(exo_vars, list) else [exo_vars]
        self.coupling_vars = copy.deepcopy(coupling_vars) if isinstance(coupling_vars, list) else [coupling_vars]
        self.refine_level = 0
        self.build_metrics = dict()     # Save refinement error metrics during training

        # Construct graph nodes
        components = [components] if not isinstance(components, list) else components
        for k, comp in enumerate(components):
            if comp['name'] == '':
                comp['name'] = f'Component {k}'
        Nk = len(components)
        nodes = {comp['name']: comp for comp in components}  # work-around since self.graph.nodes is not built yet
        for k in range(Nk):
            # Add the component as a str() node, with attributes specifying details of the surrogate
            comp_dict = components[k]
            indices, surr = self._build_component(comp_dict, nodes=nodes)
            self.graph.add_node(comp_dict['name'], surrogate=surr, is_computed=False, **indices)

        # Connect all neighbor nodes
        for node, node_obj in self.graph.nodes.items():
            for neighbor in node_obj['local_in']:
                self.graph.add_edge(neighbor, node)

        # Estimate coupling variable bounds
        if est_bds > 0:
            self._estimate_coupling_bds(est_bds)

        # Init system with most coarse fidelity indices in each component
        if init_surr:
            self.init_system()
        self.save_to_file('sys_init.pkl')

    def _build_component(self, component: ComponentSpec, nodes=None) -> tuple[dict, ComponentSurrogate]:
        """Build and return a `ComponentSurrogate` from a `dict` that describes the component model/connections.

        :param component: specifies details of a component (see `ComponentSpec`)
        :param nodes: `dict` of `{node: node_attributes}`, defaults to `self.graph.nodes`
        :returns: `connections, surr`: a `dict` of all connection indices and the `ComponentSurrogate` object
        """
        nodes = self.graph.nodes if nodes is None else nodes
        kwargs = component.get('model_kwargs', {})
        kwargs = {} if kwargs is None else kwargs

        # Set up defaults if this is a trivial one component system
        exo_in = component.get('exo_in', None)
        coupling_in = component.get('coupling_in', None)
        coupling_out = component.get('coupling_out', None)
        if len(nodes) == 1:
            exo_in = list(np.arange(0, len(self.exo_vars)))
            coupling_in = []
            coupling_out = list(np.arange(0, len(self.coupling_vars)))
        else:
            exo_in = [] if exo_in is None else exo_in
            coupling_in = [] if coupling_in is None else coupling_in
            coupling_out = [] if coupling_out is None else coupling_out
        exo_in = [exo_in] if not isinstance(exo_in, list) else exo_in
        coupling_in = [coupling_in] if not isinstance(coupling_in, list | dict) else coupling_in
        component['coupling_out'] = [coupling_out] if not isinstance(coupling_out, list) else coupling_out

        # Raise an error if all inputs or all outputs are empty
        if len(exo_in) + len(coupling_in) == 0:
            raise ValueError(f'Component {component["name"]} has no inputs! Please specify inputs in '
                             f'"exo_in" or "coupling_in".')
        if len(component['coupling_out']) == 0:
            raise ValueError(f'Component {component["name"]} has no outputs! Please specify outputs in '
                             f'"coupling_out".')

        # Get exogenous input indices (might already be a list of ints, otherwise convert list of vars to indices)
        if len(exo_in) > 0:
            if isinstance(exo_in[0], str | BaseRV):
                exo_in = [self.exo_vars.index(var) for var in exo_in]

        # Get global coupling output indices for all nodes (convert list of vars to list of indices if necessary)
        global_out = {}
        for node, node_obj in nodes.items():
            node_use = node_obj if node != component.get('name') else component
            coupling_out = node_use.get('coupling_out', None)
            coupling_out = [] if coupling_out is None else coupling_out
            coupling_out = [coupling_out] if not isinstance(coupling_out, list) else coupling_out
            global_out[node] = [self.coupling_vars.index(var) for var in coupling_out] if isinstance(
                coupling_out[0], str | BaseRV) else coupling_out

        # Refactor coupling inputs into both local and global index formats
        local_in = dict()  # e.g. {'Cathode': [0, 1, 2], 'Thruster': [0,], etc...}
        global_in = list()  # e.g. [0, 2, 4, 5, 6]
        if isinstance(coupling_in, dict):
            # If already a dict, get local connection indices from each neighbor
            for node, connections in coupling_in.items():
                conn_list = [connections] if not isinstance(connections, list) else connections
                if isinstance(conn_list[0], str | BaseRV):
                    global_ind = [self.coupling_vars.index(var) for var in conn_list]
                    local_in[node] = sorted([global_out[node].index(i) for i in global_ind])
                else:
                    local_in[node] = sorted(conn_list)

            # Convert to global coupling indices
            for node, local_idx in local_in.items():
                global_in.extend([global_out[node][i] for i in local_idx])
            global_in = sorted(global_in)
        else:
            # Otherwise, convert a list of global indices or vars into a dict of local indices
            if len(coupling_in) > 0:
                if isinstance(coupling_in[0], str | BaseRV):
                    coupling_in = [self.coupling_vars.index(var) for var in coupling_in]
            global_in = sorted(coupling_in)
            for node, node_obj in nodes.items():
                if node != component['name']:
                    l = list()
                    for i in global_in:
                        try:
                            l.append(global_out[node].index(i))
                        except ValueError:
                            pass
                    if l:
                        local_in[node] = sorted(l)

        # Store all connection indices for this component
        connections = dict(exo_in=exo_in, local_in=local_in, global_in=global_in,
                           global_out=global_out.get(component.get('name')))

        # Set up a component output save directory
        if component.get('save_output', False) and self.root_dir is not None:
            output_dir = str((Path(self.root_dir) / 'components' / component['name']).resolve())
            if not Path(output_dir).is_dir():
                os.mkdir(output_dir)
            kwargs['output_dir'] = output_dir
        else:
            if kwargs.get('output_dir', None) is not None:
                kwargs['output_dir'] = None

        # Initialize a new component surrogate
        surr_class = component.get('surrogate', 'lagrange')
        if isinstance(surr_class, str):
            match surr_class:
                case 'lagrange':
                    surr_class = SparseGridSurrogate
                case 'analytical':
                    surr_class = AnalyticalSurrogate
                case other:
                    raise NotImplementedError(f"Surrogate type '{surr_class}' is not known at this time.")

        # Check for an override of model fidelity indices (to enable just single-fidelity evaluation)
        if kwargs.get('hf_override', False):
            truth_alpha, max_alpha = (), ()
            kwargs['hf_override'] = component['truth_alpha']    # Pass in the truth alpha indices as a kwarg to model
        else:
            truth_alpha, max_alpha = component['truth_alpha'], component['max_alpha']
        max_beta = component.get('max_beta')
        truth_alpha = (truth_alpha,) if isinstance(truth_alpha, int) else truth_alpha
        max_alpha = (max_alpha,) if isinstance(max_alpha, int) else max_alpha
        max_beta = (max_beta,) if isinstance(max_beta, int) else max_beta

        # Assumes input ordering is exogenous vars + sorted coupling vars
        x_vars = [self.exo_vars[i] for i in exo_in] + [self.coupling_vars[i] for i in global_in]
        surr = surr_class(x_vars, component['model'], truth_alpha=truth_alpha, max_alpha=max_alpha,
                          max_beta=max_beta, executor=self.executor, log_file=self.log_file,
                          model_args=component.get('model_args'), model_kwargs=kwargs)
        return connections, surr

    def swap_component(self, component: ComponentSpec, exo_add: BaseRV | list[BaseRV] = None,
                       exo_remove: IndicesRV = None, qoi_add: BaseRV | list[BaseRV] = None,
                       qoi_remove: IndicesRV = None):
        """Swap a new component into the system, updating all connections/inputs.

        !!! Warning "Beta feature, proceed with caution"
            If you are swapping a new component in, you cannot remove any inputs that are expected by other components,
            including the coupling variables output by the current model.

        :param component: specs of new component model (must replace an existing component with matching `name`)
        :param exo_add: variables to add to system exogenous inputs (will be appended to end)
        :param exo_remove: indices of system exogenous inputs to delete (can't be shared by other components)
        :param qoi_add: system output QoIs to add (will be appended to end of `coupling_vars`)
        :param qoi_remove: indices of system `coupling_vars` to delete (can't be shared by other components)
        """
        # Delete system exogenous inputs
        if exo_remove is None:
            exo_remove = []
        exo_remove = [exo_remove] if not isinstance(exo_remove, list) else exo_remove
        exo_remove = [self.exo_vars.index(var) for var in exo_remove] if exo_remove and isinstance(
            exo_remove[0], str | BaseRV) else exo_remove

        exo_remove = sorted(exo_remove)
        for j, exo_var_idx in enumerate(exo_remove):
            # Adjust exogenous indices for all components to account for deleted system inputs
            for node, node_obj in self.graph.nodes.items():
                if node != component['name']:
                    for i, idx in enumerate(node_obj['exo_in']):
                        if idx == exo_var_idx:
                            error_msg = f"Can't delete system exogenous input at idx {exo_var_idx}, since it is " \
                                        f"shared by component '{node}'."
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                        if idx > exo_var_idx:
                            node_obj['exo_in'][i] -= 1

            # Need to update the remaining delete indices by -1 to account for each sequential deletion
            del self.exo_vars[exo_var_idx]
            for i in range(j+1, len(exo_remove)):
                exo_remove[i] -= 1

        # Append any new exogenous inputs to the end
        if exo_add is not None:
            exo_add = [exo_add] if not isinstance(exo_add, list) else exo_add
            self.exo_vars.extend(exo_add)

        # Delete system qoi outputs (if not shared by other components)
        qoi_remove = sorted(self._get_qoi_ind(qoi_remove))
        for j, qoi_idx in enumerate(qoi_remove):
            # Adjust coupling indices for all components to account for deleted system outputs
            for node, node_obj in self.graph.nodes.items():
                if node != component['name']:
                    for i, idx in enumerate(node_obj['global_in']):
                        if idx == qoi_idx:
                            error_msg = f"Can't delete system QoI at idx {qoi_idx}, since it is an input to " \
                                        f"component '{node}'."
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                        if idx > qoi_idx:
                            node_obj['global_in'][i] -= 1

                    for i, idx in enumerate(node_obj['global_out']):
                        if idx > qoi_idx:
                            node_obj['global_out'][i] -= 1

            # Need to update the remaining delete indices by -1 to account for each sequential deletion
            del self.coupling_vars[qoi_idx]
            for i in range(j+1, len(qoi_remove)):
                qoi_remove[i] -= 1

        # Append any new system QoI outputs to the end
        if qoi_add is not None:
            qoi_add = [qoi_add] if not isinstance(qoi_add, list) else qoi_add
            self.coupling_vars.extend(qoi_add)

        # Build and initialize the new component surrogate
        indices, surr = self._build_component(component)
        surr.init_coarse()

        # Make changes to adj matrix if coupling inputs changed
        prev_neighbors = list(self.graph.nodes[component['name']]['local_in'].keys())
        new_neighbors = list(indices['local_in'].keys())
        for neighbor in new_neighbors:
            if neighbor not in prev_neighbors:
                self.graph.add_edge(neighbor, component['name'])
            else:
                prev_neighbors.remove(neighbor)
        for neighbor in prev_neighbors:
            self.graph.remove_edge(neighbor, component['name'])

        self.logger.info(f"Swapped component '{component['name']}'.")
        nx.set_node_attributes(self.graph, {component['name']: {'exo_in': indices['exo_in'], 'local_in':
                                                                indices['local_in'], 'global_in': indices['global_in'],
                                                                'global_out': indices['global_out'],
                                                                'surrogate': surr, 'is_computed': False}})

    def insert_component(self, component: ComponentSpec, exo_add: BaseRV | list[BaseRV] = None,
                         qoi_add: BaseRV | list[BaseRV] = None):
        """Insert a new component into the system.

        :param component: specs of new component model
        :param exo_add: variables to add to system exogenous inputs (will be appended to end of `exo_vars`)
        :param qoi_add: system output QoIs to add (will be appended to end of `coupling_vars`)
        """
        if exo_add is not None:
            exo_add = [exo_add] if not isinstance(exo_add, list) else exo_add
            self.exo_vars.extend(exo_add)
        if qoi_add is not None:
            qoi_add = [qoi_add] if not isinstance(qoi_add, list) else qoi_add
            self.coupling_vars.extend(qoi_add)

        indices, surr = self._build_component(component)
        surr.init_coarse()
        self.graph.add_node(component['name'], surrogate=surr, is_computed=False, **indices)

        # Add graph edges
        neighbors = list(indices['local_in'].keys())
        for neighbor in neighbors:
            self.graph.add_edge(neighbor, component['name'])
        self.logger.info(f"Inserted component '{component['name']}'.")

    def _save_on_error(func):
        """Gracefully exit and save `SystemSurrogate` on any errors."""
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except:
                self.save_to_file('sys_error.pkl')
                self.logger.critical(f'An error occurred during execution of {func.__name__}. Saving '
                                     f'SystemSurrogate object to sys_error.pkl', exc_info=True)
                self.logger.info(f'Final system surrogate on exit: \n {self}')
                raise
        return wrap
    _save_on_error = staticmethod(_save_on_error)

    @_save_on_error
    def init_system(self):
        """Add the coarsest multi-index to each component surrogate."""
        self._print_title_str('Initializing all component surrogates')
        for node, node_obj in self.graph.nodes.items():
            node_obj['surrogate'].init_coarse()
            # for alpha, beta in list(node_obj['surrogate'].candidate_set):
            #     # Add one refinement in each input dimension to initialize
            #     node_obj['surrogate'].activate_index(alpha, beta)
            self.logger.info(f"Initialized component '{node}'.")

    @_save_on_error
    def fit(self, qoi_ind: IndicesRV = None, num_refine: int = 100, max_iter: int = 20, max_tol: float = 1e-3,
            max_runtime: float = 1, save_interval: int = 0, update_bounds: bool = True, test_set: dict = None,
            n_jobs: int = 1):
        """Train the system surrogate adaptively by iterative refinement until an end condition is met.

        :param qoi_ind: list of system QoI variables to focus refinement on, use all QoI if not specified
        :param num_refine: number of samples of exogenous inputs to compute error indicators on
        :param max_iter: the maximum number of refinement steps to take
        :param max_tol: the max allowable value in relative L2 error to achieve
        :param max_runtime: the maximum wall clock time (hr) to run refinement for (will go until all models finish)
        :param save_interval: number of refinement steps between each progress save, none if 0
        :param update_bounds: whether to continuously update coupling variable bounds during refinement
        :param test_set: `dict(xt=(Nt, x_dim), yt=(Nt, y_dim)` to show convergence of surrogate to the truth model
        :param n_jobs: number of cpu workers for computing error indicators (on master MPI task), 1=sequential
        """
        qoi_ind = self._get_qoi_ind(qoi_ind)
        Nqoi = len(qoi_ind)
        max_iter = self.refine_level + max_iter
        curr_error = np.inf
        t_start = time.time()
        test_stats, xt, yt, t_fig, t_ax = None, None, None, None, None

        # Record of (error indicator, component, alpha, beta, num_evals, total added cost (s)) for each iteration
        train_record = self.build_metrics.get('train_record', [])
        if test_set is not None:
            xt, yt = test_set['xt'], test_set['yt']
        xt, yt = self.build_metrics.get('xt', xt), self.build_metrics.get('yt', yt)  # Overrides test set param

        # Track convergence progress on a test set and on the max error indicator
        err_fig, err_ax = plt.subplots()
        if xt is not None and yt is not None:
            self.build_metrics['xt'] = xt
            self.build_metrics['yt'] = yt
            if self.build_metrics.get('test_stats') is not None:
                test_stats = self.build_metrics.get('test_stats')
            else:
                # Get initial perf metrics, (2, Nqoi)
                test_stats = np.expand_dims(self.get_test_metrics(xt, yt, qoi_ind=qoi_ind), axis=0)
            t_fig, t_ax = plt.subplots(1, Nqoi) if Nqoi > 1 else plt.subplots()

        # Set up a parallel pool of workers, sequential if n_jobs=1
        with Parallel(n_jobs=n_jobs, verbose=0) as ppool:
            while True:
                # Check all end conditions
                if self.refine_level >= max_iter:
                    self._print_title_str(f'Termination criteria reached: Max iteration {self.refine_level}/{max_iter}')
                    break
                if curr_error == -np.inf:
                    self._print_title_str(f'Termination criteria reached: No candidates left to refine')
                    break
                if curr_error < max_tol:
                    self._print_title_str(f'Termination criteria reached: L2 error {curr_error} < tol {max_tol}')
                    break
                if ((time.time() - t_start)/3600.0) >= max_runtime:
                    actual = datetime.timedelta(seconds=time.time()-t_start)
                    target = datetime.timedelta(seconds=max_runtime*3600)
                    self._print_title_str(f'Termination criteria reached: runtime {str(actual)} > {str(target)}')
                    break

                # Refine surrogate and save progress
                refine_res = self.refine(qoi_ind=qoi_ind, num_refine=num_refine, update_bounds=update_bounds,
                                         ppool=ppool)
                curr_error = refine_res[0]
                if save_interval > 0 and self.refine_level % save_interval == 0:
                    self.save_to_file(f'sys_iter_{self.refine_level}.pkl')

                # Plot progress of error indicator
                train_record.append(refine_res)
                error_record = [res[0] for res in train_record]
                self.build_metrics['train_record'] = train_record
                err_ax.clear(); err_ax.grid(); err_ax.plot(error_record, '-k')
                ax_default(err_ax, 'Iteration', r'Relative $L_2$ error indicator', legend=False)
                err_ax.set_yscale('log')
                if self.root_dir is not None:
                    err_fig.savefig(str(Path(self.root_dir) / 'error_indicator.png'), dpi=300, format='png')

                # Plot progress on test set
                if xt is not None and yt is not None:
                    stats = self.get_test_metrics(xt, yt, qoi_ind=qoi_ind)
                    test_stats = np.concatenate((test_stats, stats[np.newaxis, ...]), axis=0)
                    for i in range(Nqoi):
                        ax = t_ax if Nqoi == 1 else t_ax[i]
                        ax.clear(); ax.grid(); ax.set_yscale('log')
                        ax.plot(test_stats[:, 1, i], '-k')
                        ax.set_title(self.coupling_vars[qoi_ind[i]].to_tex(units=True))
                        ax_default(ax, 'Iteration', r'Relative $L_2$ error', legend=False)
                    t_fig.set_size_inches(3.5*Nqoi, 3.5)
                    t_fig.tight_layout()
                    if self.root_dir is not None:
                        t_fig.savefig(str(Path(self.root_dir) / 'test_set.png'), dpi=300, format='png')
                    self.build_metrics['test_stats'] = test_stats

        self.save_to_file(f'sys_final.pkl')
        self.logger.info(f'Final system surrogate: \n {self}')

    def get_allocation(self, idx: int = None):
        """Get a breakdown of cost allocation up to a certain iteration number during training (starting at 1).

        :param idx: the iteration number to get allocation results for (defaults to last refinement step)
        :returns: `cost_alloc, offline_alloc, cost_cum` - the cost alloc per node/fidelity and cumulative training cost
        """
        if idx is None:
            idx = self.refine_level
        if idx > self.refine_level:
            raise ValueError(f'Specified index: {idx} is greater than the max training level of {self.refine_level}')

        cost_alloc = dict()     # Cost allocation per node and model fidelity
        cost_cum = [0.0]        # Cumulative cost allocation during training

        # Add initialization costs for each node
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            base_alpha = (0,) * len(surr.truth_alpha)
            base_beta = (0,) * (len(surr.max_refine) - len(surr.truth_alpha))
            base_cost = surr.get_cost(base_alpha, base_beta)
            cost_alloc[node] = dict()
            if base_cost > 0:
                cost_alloc[node][str(base_alpha)] = np.array([1, float(base_cost)])
                cost_cum[0] += float(base_cost)

        # Add cumulative training costs
        for i in range(idx):
            err_indicator, node, alpha, beta, num_evals, cost = self.build_metrics['train_record'][i]
            if cost_alloc[node].get(str(alpha), None) is None:
                cost_alloc[node][str(alpha)] = np.zeros(2)  # (num model evals, total cpu_time cost)
            cost_alloc[node][str(alpha)] += [round(num_evals), float(cost)]
            cost_cum.append(float(cost))

        # Get summary of total offline costs spent building search candidates (i.e. training overhead)
        offline_alloc = dict()
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            offline_alloc[node] = dict()
            for alpha, beta in surr.candidate_set:
                if offline_alloc[node].get(str(alpha), None) is None:
                    offline_alloc[node][str(alpha)] = np.zeros(2)   # (num model evals, total cpu_time cost)
                added_cost = surr.get_cost(alpha, beta)
                base_cost = surr.get_sub_surrogate(alpha, beta).model_cost
                offline_alloc[node][str(alpha)] += [round(added_cost/base_cost), float(added_cost)]

        return cost_alloc, offline_alloc, np.cumsum(cost_cum)

    def get_test_metrics(self, xt: np.ndarray, yt: np.ndarray, qoi_ind: IndicesRV = None,
                         training: bool = True) -> np.ndarray:
        """Get relative L2 error metric over a test set.

        :param xt: `(Nt, x_dim)` random test set of inputs
        :param yt: `(Nt, y_dim)` random test set outputs
        :param qoi_ind: list of indices of QoIs to get metrics for
        :param training: whether to evaluate the surrogate in training or evaluation mode
        :returns: `stats` - `(2, Nqoi)` array &rarr; `[num_candidates, rel_L2_error]` for each QoI
        """
        qoi_ind = self._get_qoi_ind(qoi_ind)
        ysurr = self(xt, training=training)
        ysurr = ysurr[:, qoi_ind]
        yt = yt[:, qoi_ind]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_l2_err = np.sqrt(np.mean((yt - ysurr) ** 2, axis=0)) / np.sqrt(np.mean(yt ** 2, axis=0))
            rel_l2_err = np.nan_to_num(rel_l2_err, posinf=np.nan, neginf=np.nan, nan=np.nan)
        num_cands = 0
        for node, node_obj in self.graph.nodes.items():
            num_cands += len(node_obj['surrogate'].index_set) + len(node_obj['surrogate'].candidate_set)

        # Get test stats for each QoI
        stats = np.zeros((2, yt.shape[-1]))
        self.logger.debug(f'{"QoI idx": >10} {"Iteration": >10} {"len(I_k)": >10} {"Relative L2": >15}')
        for i in range(yt.shape[-1]):
            stats[:, i] = np.array([num_cands, rel_l2_err[i]])
            self.logger.debug(f'{i: 10d} {self.refine_level: 10d} {num_cands: 10d} {rel_l2_err[i]: 15.5f}')

        return stats

    def _get_qoi_ind(self, qoi_ind: IndicesRV) -> list[int]:
        """Small helper to make sure QoI indices are a list of integers."""
        if qoi_ind is None:
            qoi_ind = list(np.arange(0, len(self.coupling_vars)))
        qoi_ind = [qoi_ind] if not isinstance(qoi_ind, list) else qoi_ind
        qoi_ind = [self.coupling_vars.index(var) for var in qoi_ind] if qoi_ind and isinstance(
            qoi_ind[0], str | BaseRV) else qoi_ind

        return qoi_ind

    def refine(self, qoi_ind: IndicesRV = None, num_refine: int = 100, update_bounds: bool = True,
               ppool: Parallel = None) -> tuple:
        """Find and refine the component surrogate with the largest error on system-level QoI.

        :param qoi_ind: indices of system QoI to focus surrogate refinement on, use all QoI if not specified
        :param num_refine: number of samples of exogenous inputs to compute error indicators on
        :param update_bounds: whether to continuously update coupling variable bounds
        :param ppool: a `Parallel` instance from `joblib` to compute error indicators in parallel, None=sequential
        :returns refine_res: a tuple of `(error_indicator, component, node_star, alpha_star, beta_star, N, cost)`
                             indicating the chosen candidate index and incurred cost
        """
        self._print_title_str(f'Refining system surrogate: iteration {self.refine_level + 1}')
        set_loky_pickler('dill')    # Dill can serialize 'self' for parallel workers
        temp_exc = self.executor    # It can't serialize an executor though, so must save this temporarily
        self.set_executor(None)
        qoi_ind = self._get_qoi_ind(qoi_ind)

        # Compute entire integrated-surrogate on a random test set for global system QoI error estimation
        x_exo = self.sample_inputs((num_refine,))
        y_curr = self(x_exo, training=True)
        y_min, y_max = None, None
        if update_bounds:
            y_min = np.min(y_curr, axis=0, keepdims=True)  # (1, ydim)
            y_max = np.max(y_curr, axis=0, keepdims=True)  # (1, ydim)

        # Find the candidate surrogate with the largest error indicator
        error_max, error_indicator = -np.inf, -np.inf
        node_star, alpha_star, beta_star, l2_star, cost_star = None, None, None, -np.inf, 0
        for node, node_obj in self.graph.nodes.items():
            self.logger.info(f"Estimating error for component '{node}'...")
            candidates = node_obj['surrogate'].candidate_set.copy()

            def compute_error(alpha, beta):
                # Helper function for computing error indicators for a given candidate (alpha, beta)
                index_set = node_obj['surrogate'].index_set.copy()
                index_set.append((alpha, beta))
                y_cand = self(x_exo, training=True, index_set={node: index_set})
                ymin = np.min(y_cand, axis=0, keepdims=True)
                ymax = np.max(y_cand, axis=0, keepdims=True)
                error = y_cand[:, qoi_ind] - y_curr[:, qoi_ind]
                rel_l2 = np.sqrt(np.nanmean(error ** 2, axis=0)) / np.sqrt(np.nanmean(y_cand[:, qoi_ind] ** 2, axis=0))
                rel_l2 = np.nan_to_num(rel_l2, nan=np.nan, posinf=np.nan, neginf=np.nan)
                delta_error = np.nanmax(rel_l2)  # Max relative L2 error over all system QoIs
                delta_work = max(1, node_obj['surrogate'].get_cost(alpha, beta))  # Cpu time (s)

                return ymin, ymax, delta_error, delta_work

            if len(candidates) > 0:
                ret = ppool(delayed(compute_error)(alpha, beta) for alpha, beta in candidates) if ppool is not None \
                    else [compute_error(alpha, beta) for alpha, beta in candidates]

                for i, (ymin, ymax, d_error, d_work) in enumerate(ret):
                    if update_bounds:
                        y_min = np.min(np.concatenate((y_min, ymin), axis=0), axis=0, keepdims=True)
                        y_max = np.max(np.concatenate((y_max, ymax), axis=0), axis=0, keepdims=True)
                    alpha, beta = candidates[i]
                    error_indicator = d_error / d_work
                    self.logger.info(f"Candidate multi-index: {(alpha, beta)}. Error indicator: "
                                     f"{error_indicator}. L2 error: {d_error}")

                    if error_indicator > error_max:
                        error_max = error_indicator
                        node_star, alpha_star, beta_star, l2_star, cost_star = node, alpha, beta, d_error, d_work
            else:
                self.logger.info(f"Component '{node}' has no available candidates left!")

        # Update all coupling variable ranges
        if update_bounds:
            for i in range(y_curr.shape[-1]):
                self._update_coupling_bds(i, (y_min[0, i], y_max[0, i]))

        # Add the chosen multi-index to the chosen component
        self.set_executor(temp_exc)
        if node_star is not None:
            self.logger.info(f"Candidate multi-index {(alpha_star, beta_star)} chosen for component '{node_star}'")
            self.graph.nodes[node_star]['surrogate'].activate_index(alpha_star, beta_star)
            self.refine_level += 1
            num_evals = round(cost_star / self[node_star].get_sub_surrogate(alpha_star, beta_star).model_cost)
        else:
            self.logger.info(f"No candidates left for refinement, iteration: {self.refine_level}")
            num_evals = 0

        return l2_star, node_star, alpha_star, beta_star, num_evals, cost_star

    def predict(self, x: np.ndarray | float, max_fpi_iter: int = 100, anderson_mem: int = 10, fpi_tol: float = 1e-10,
                use_model: str | tuple | dict = None, model_dir: str | Path = None, verbose: bool = False,
                training: bool = False, index_set: dict[str: IndexSet] = None, qoi_ind: IndicesRV = None) -> np.ndarray:
        """Evaluate the system surrogate at exogenous inputs `x`.

        !!! Warning
            You can use this function to predict outputs for your MD system using the full-order models rather than the
            surrogate, by specifying `use_model`. This is convenient because `SystemSurrogate` manages all the
            coupled information flow between models automatically. However, it is *highly* recommended to not use
            the full model if your system contains feedback loops. The FPI nonlinear solver would be infeasible using
            anything more computationally demanding than the surrogate.

        :param x: `(..., x_dim)` the points to get surrogate predictions for
        :param max_fpi_iter: the limit on convergence for the fixed-point iteration routine
        :param anderson_mem: hyperparameter for tuning the convergence of FPI with anderson acceleration
        :param fpi_tol: tolerance limit for convergence of fixed-point iteration
        :param use_model: 'best'=highest-fidelity, 'worst'=lowest-fidelity, tuple=specific fidelity, None=surrogate,
                           specify a `dict` of the above to assign different model fidelities for diff components
        :param model_dir: directory to save model outputs if `use_model` is specified
        :param verbose: whether to print out iteration progress during execution
        :param training: whether to call the system surrogate in training or evaluation mode, ignored if `use_model`
        :param index_set: `dict(node=[indices])` to override default index set for a node (only useful for parallel)
        :param qoi_ind: list of qoi indices to return, defaults to returning all system `coupling_vars`
        :returns y: `(..., y_dim)` the surrogate approximation of the system QoIs
        """
        # Allocate space for all system outputs (just save all coupling vars)
        x = np.atleast_1d(x)
        ydim = len(self.coupling_vars)
        y = np.zeros(x.shape[:-1] + (ydim,))
        valid_idx = ~np.isnan(x[..., 0])  # Keep track of valid samples (set to False if FPI fails)
        t1 = 0
        output_dir = None

        # Interpret which model fidelities to use for each component (if specified)
        if use_model is not None:
            if not isinstance(use_model, dict):
                use_model = {node: use_model for node in self.graph.nodes}  # use same for each component
        else:
            use_model = {node: None for node in self.graph.nodes}
        use_model = {node: use_model.get(node, None) for node in self.graph.nodes}

        # Initialize all components
        for node, node_obj in self.graph.nodes.items():
            node_obj['is_computed'] = False

        # Convert system into DAG by grouping strongly-connected-components
        dag = nx.condensation(self.graph)

        # Compute component models in topological order
        for supernode in nx.topological_sort(dag):
            scc = [n for n in dag.nodes[supernode]['members']]

            # Compute single component feedforward output (no FPI needed)
            if len(scc) == 1:
                if verbose:
                    self.logger.info(f"Running component '{scc[0]}'...")
                    t1 = time.time()

                # Gather inputs
                node_obj = self.graph.nodes[scc[0]]
                exo_inputs = x[..., node_obj['exo_in']]
                # for comp_name in node_obj['local_in']:
                #     assert self.graph.nodes[comp_name]['is_computed']
                coupling_inputs = y[..., node_obj['global_in']]
                comp_input = np.concatenate((exo_inputs, coupling_inputs), axis=-1)  # (..., xdim)

                # Compute outputs
                indices = index_set.get(scc[0], None) if index_set is not None else None
                if model_dir is not None:
                    output_dir = Path(model_dir) / scc[0]
                    os.mkdir(output_dir)
                comp_output = node_obj['surrogate'](comp_input[valid_idx, :], use_model=use_model.get(scc[0]),
                                                    model_dir=output_dir, training=training, index_set=indices)
                for local_i, global_i in enumerate(node_obj['global_out']):
                    y[valid_idx, global_i] = comp_output[..., local_i]
                node_obj['is_computed'] = True

                if verbose:
                    self.logger.info(f"Component '{scc[0]}' completed. Runtime: {time.time() - t1} s")

            # Handle FPI for SCCs with more than one component
            else:
                # Set the initial guess for all coupling vars (middle of domain)
                coupling_bds = [rv.bounds() for rv in self.coupling_vars]
                x_couple = np.array([(bds[0] + bds[1]) / 2 for bds in coupling_bds])
                x_couple = np.broadcast_to(x_couple, x.shape[:-1] + x_couple.shape).copy()

                adj_nodes = []
                fpi_idx = set()
                for node in scc:
                    for comp_name, local_idx in self.graph.nodes[node]['local_in'].items():
                        # Track the global idx of all coupling vars that need FPI
                        if comp_name in scc:
                            fpi_idx.update([self.graph.nodes[comp_name]['global_out'][idx] for idx in local_idx])

                        # Override coupling vars from components outside the scc (should already be computed)
                        if comp_name not in scc and comp_name not in adj_nodes:
                            # assert self.graph.nodes[comp_name]['is_computed']
                            global_idx = self.graph.nodes[comp_name]['global_out']
                            x_couple[..., global_idx] = y[..., global_idx]
                            adj_nodes.append(comp_name)  # Only need to do this once for each adj component
                x_couple_next = x_couple.copy()
                fpi_idx = sorted(fpi_idx)

                # Main FPI loop
                if verbose:
                    self.logger.info(f"Initializing FPI for SCC {scc} ...")
                    t1 = time.time()
                k = 0
                residual_hist = None
                x_hist = None
                while True:
                    for node in scc:
                        # Gather inputs from exogenous and coupling sources
                        node_obj = self.graph.nodes[node]
                        exo_inputs = x[..., node_obj['exo_in']]
                        coupling_inputs = x_couple[..., node_obj['global_in']]
                        comp_input = np.concatenate((exo_inputs, coupling_inputs), axis=-1)     # (..., xdim)

                        # Compute component outputs (just don't do this FPI with the real models, please..)
                        indices = index_set.get(node, None) if index_set is not None else None
                        comp_output = node_obj['surrogate'](comp_input[valid_idx, :], use_model=use_model.get(node),
                                                            model_dir=None, training=training, index_set=indices)
                        global_idx = node_obj['global_out']
                        for local_i, global_i in enumerate(global_idx):
                            x_couple_next[valid_idx, global_i] = comp_output[..., local_i]
                            # Can't splice valid_idx with global_idx for some reason, have to loop over global_idx here

                    # Compute residual and check end conditions
                    residual = np.expand_dims(x_couple_next[..., fpi_idx] - x_couple[..., fpi_idx], axis=-1)
                    max_error = np.max(np.abs(residual[valid_idx, :, :]))
                    if verbose:
                        self.logger.info(f'FPI iter: {k}. Max residual: {max_error}. Time: {time.time() - t1} s')
                    if max_error <= fpi_tol:
                        if verbose:
                            self.logger.info(f'FPI converged for SCC {scc} in {k} iterations with {max_error} < tol '
                                             f'{fpi_tol}. Final time: {time.time() - t1} s')
                        break
                    if k >= max_fpi_iter:
                        self.logger.warning(f'FPI did not converge in {max_fpi_iter} iterations for SCC {scc}: '
                                            f'{max_error} > tol {fpi_tol}. Some samples will be returned as NaN.')
                        converged_idx = np.max(np.abs(residual), axis=(-1, -2)) <= fpi_tol
                        for idx in fpi_idx:
                            y[~converged_idx, idx] = np.nan
                        valid_idx = np.logical_and(valid_idx, converged_idx)
                        break

                    # Keep track of residual and x_couple histories
                    if k == 0:
                        residual_hist = residual.copy()                                 # (..., xdim, 1)
                        x_hist = np.expand_dims(x_couple_next[..., fpi_idx], axis=-1)   # (..., xdim, 1)
                        x_couple[:] = x_couple_next[:]
                        k += 1
                        continue  # skip anderson accel on first iteration

                    # Iterate with anderson acceleration (only iterate on samples that are not yet converged)
                    converged_idx = np.max(np.abs(residual), axis=(-1, -2)) <= fpi_tol
                    curr_idx = np.logical_and(valid_idx, ~converged_idx)
                    residual_hist = np.concatenate((residual_hist, residual), axis=-1)
                    x_hist = np.concatenate((x_hist, np.expand_dims(x_couple_next[..., fpi_idx], axis=-1)), axis=-1)
                    mk = min(anderson_mem, k)
                    Fk = residual_hist[curr_idx, :, k-mk:]                               # (..., xdim, mk+1)
                    C = np.ones(Fk.shape[:-2] + (1, mk + 1))
                    b = np.zeros(Fk.shape[:-2] + (len(fpi_idx), 1))
                    d = np.ones(Fk.shape[:-2] + (1, 1))
                    alpha = np.expand_dims(self._constrained_lls(Fk, b, C, d), axis=-3)   # (..., 1, mk+1, 1)
                    x_new = np.squeeze(x_hist[curr_idx, :, np.newaxis, -(mk+1):] @ alpha, axis=(-1, -2))
                    for local_i, global_i in enumerate(fpi_idx):
                        x_couple[curr_idx, global_i] = x_new[..., local_i]
                    k += 1

                # Save outputs of each component in SCC after convergence of FPI
                for node in scc:
                    global_idx = self.graph.nodes[node]['global_out']
                    for global_i in global_idx:
                        y[valid_idx, global_i] = x_couple_next[valid_idx, global_i]
                    self.graph.nodes[node]['is_computed'] = True

        # Return all component outputs (..., Nqoi), samples that didn't converge during FPI are left as np.nan
        qoi_ind = self._get_qoi_ind(qoi_ind)
        return y[..., qoi_ind]

    def __call__(self, *args, **kwargs):
        """Convenience wrapper to allow calling as `ret = SystemSurrogate(x)`."""
        return self.predict(*args, **kwargs)

    def _estimate_coupling_bds(self, num_est: int, anderson_mem: int = 10, fpi_tol: float = 1e-10,
                               max_fpi_iter: int = 100):
        """Estimate and set the coupling variable bounds.

        :param num_est: the number of samples of exogenous inputs to use
        :param anderson_mem: FPI hyperparameter (default is usually good)
        :param fpi_tol: floating point tolerance for FPI convergence
        :param max_fpi_iter: maximum number of FPI iterations
        """
        self._print_title_str('Estimating coupling variable bounds')
        x = self.sample_inputs((num_est,))
        y = self(x, use_model='best', verbose=True, anderson_mem=anderson_mem, fpi_tol=fpi_tol,
                 max_fpi_iter=max_fpi_iter)
        for i in range(len(self.coupling_vars)):
            lb = np.nanmin(y[:, i])
            ub = np.nanmax(y[:, i])
            self._update_coupling_bds(i, (lb, ub), init=True)

    def _update_coupling_bds(self, global_idx: int, bds: tuple, init: bool = False, buffer: float = 0.05):
        """Update coupling variable bounds.

        :param global_idx: global index of coupling variable to update
        :param bds: new bounds to update the current bounds with
        :param init: whether to set new bounds or update existing (default)
        :param buffer: fraction of domain length to buffer upper/lower bounds
        """
        offset = buffer * (bds[1] - bds[0])
        offset_bds = (bds[0] - offset, bds[1] + offset)
        coupling_bds = [rv.bounds() for rv in self.coupling_vars]
        new_bds = offset_bds if init else (min(coupling_bds[global_idx][0], offset_bds[0]),
                                           max(coupling_bds[global_idx][1], offset_bds[1]))
        self.coupling_vars[global_idx].update_bounds(*new_bds)

        # Iterate over all components and update internal coupling variable bounds
        for node_name, node_obj in self.graph.nodes.items():
            if global_idx in node_obj['global_in']:
                # Get the local index for this coupling variable within each component's inputs
                local_idx = len(node_obj['exo_in']) + node_obj['global_in'].index(global_idx)
                node_obj['surrogate'].update_input_bds(local_idx, new_bds)

    def sample_inputs(self, size: tuple | int, comp: str = 'System', use_pdf: bool = False,
                      nominal: dict[str: float] = None, constants: set[str] = None) -> np.ndarray:
        """Return samples of the inputs according to provided options.

        :param size: tuple or integer specifying shape or number of samples to obtain
        :param comp: which component to sample inputs for (defaults to full system exogenous inputs)
        :param use_pdf: whether to sample from each variable's pdf, defaults to random samples over input domain instead
        :param nominal: `dict(var_id=value)` of nominal values for params with relative uncertainty, also can use
                        to specify constant values for a variable listed in `constants`
        :param constants: set of param types to hold constant while sampling (i.e. calibration, design, etc.),
                          can also put a `var_id` string in here to specify a single variable to hold constant
        :returns x: `(*size, x_dim)` samples of the inputs for the given component/system
        """
        size = (size, ) if isinstance(size, int) else size
        if nominal is None:
            nominal = dict()
        if constants is None:
            constants = set()
        x_vars = self.exo_vars if comp == 'System' else self[comp].x_vars
        x = np.empty((*size, len(x_vars)))
        for i, var in enumerate(x_vars):
            # Set a constant value for this variable
            if var.param_type in constants or var in constants:
                x[..., i] = nominal.get(var, var.nominal)  # Defaults to variable's nominal value if not specified

            # Sample from this variable's pdf or randomly within its domain bounds (reject if outside bounds)
            else:
                lb, ub = var.bounds()
                x_sample = var.sample(size, nominal=nominal.get(var, None)) if use_pdf \
                    else var.sample_domain(size)
                good_idx = (x_sample < ub) & (x_sample > lb)
                num_reject = np.sum(~good_idx)

                while num_reject > 0:
                    new_sample = var.sample((num_reject,), nominal=nominal.get(var, None)) if use_pdf \
                        else var.sample_domain((num_reject,))
                    x_sample[~good_idx] = new_sample
                    good_idx = (x_sample < ub) & (x_sample > lb)
                    num_reject = np.sum(~good_idx)

                x[..., i] = x_sample

        return x

    def plot_slice(self, slice_idx: IndicesRV = None, qoi_idx: IndicesRV = None, show_surr: bool = True,
                   show_model: list = None, model_dir: str | Path = None, N: int = 50, nominal: dict[str: float] = None,
                   random_walk: bool = False, from_file: str | Path = None):
        """Helper function to plot 1d slices of the surrogate and/or model(s) over the inputs.

        :param slice_idx: list of exogenous input variables or indices to take 1d slices of
        :param qoi_idx: list of model output variables or indices to plot 1d slices of
        :param show_surr: whether to show the surrogate prediction
        :param show_model: also plot model predictions, list() of ['best', 'worst', tuple(alpha), etc.]
        :param model_dir: base directory to save model outputs (if specified)
        :param N: the number of points to take in the 1d slice
        :param nominal: `dict` of `str(var)->nominal` to use as constant value for all non-sliced variables
        :param random_walk: whether to slice in a random d-dimensional direction or hold all params const while slicing
        :param from_file: path to a .pkl file to load a saved slice from disk
        :returns: `fig, ax` with `num_slice` by `num_qoi` subplots
        """
        # Manage loading important quantities from file (if provided)
        xs, ys_model, ys_surr = None, None, None
        if from_file is not None:
            with open(Path(from_file), 'rb') as fd:
                slice_data = pickle.load(fd)
                slice_idx = slice_data['slice_idx']     # Must use same input slices as save file
                show_model = slice_data['show_model']   # Must use same model data as save file
                qoi_idx = slice_data['qoi_idx'] if qoi_idx is None else qoi_idx
                xs = slice_data['xs']
                model_dir = None  # Don't run or save any models if loading from file

        # Set default values (take up to the first 3 slices by default)
        rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        if model_dir is not None:
            os.mkdir(Path(model_dir) / f'sweep_{rand_id}')
        if nominal is None:
            nominal = dict()
        slice_idx = list(np.arange(0, min(3, len(self.exo_vars)))) if slice_idx is None else slice_idx
        qoi_idx = list(np.arange(0, min(3, len(self.coupling_vars)))) if qoi_idx is None else qoi_idx
        if isinstance(slice_idx[0], str | BaseRV):
            slice_idx = [self.exo_vars.index(var) for var in slice_idx]
        if isinstance(qoi_idx[0], str | BaseRV):
            qoi_idx = [self.coupling_vars.index(var) for var in qoi_idx]

        exo_bds = [var.bounds() for var in self.exo_vars]
        xlabels = [self.exo_vars[idx].to_tex(units=True) for idx in slice_idx]
        ylabels = [self.coupling_vars[idx].to_tex(units=True) for idx in qoi_idx]

        # Construct slice model inputs (if not provided)
        if xs is None:
            xs = np.zeros((N, len(slice_idx), len(self.exo_vars)))
            for i in range(len(slice_idx)):
                if random_walk:
                    # Make a random straight-line walk across d-cube
                    r0 = np.squeeze(self.sample_inputs((1,), use_pdf=False), axis=0)
                    r0[slice_idx[i]] = exo_bds[slice_idx[i]][0]             # Start slice at this lower bound
                    rf = np.squeeze(self.sample_inputs((1,), use_pdf=False), axis=0)
                    rf[slice_idx[i]] = exo_bds[slice_idx[i]][1]             # Slice up to this upper bound
                    xs[0, i, :] = r0
                    for k in range(1, N):
                        xs[k, i, :] = xs[k-1, i, :] + (rf-r0)/(N-1)
                else:
                    # Otherwise, only slice one variable
                    for j in range(len(self.exo_vars)):
                        if j == slice_idx[i]:
                            xs[:, i, j] = np.linspace(exo_bds[slice_idx[i]][0], exo_bds[slice_idx[i]][1], N)
                        else:
                            xs[:, i, j] = nominal.get(self.exo_vars[j], self.exo_vars[j].nominal)

        # Walk through each model that is requested by show_model
        if show_model is not None:
            if from_file is not None:
                ys_model = slice_data['ys_model']
            else:
                ys_model = list()
                for model in show_model:
                    output_dir = None
                    if model_dir is not None:
                        output_dir = (Path(model_dir) / f'sweep_{rand_id}' /
                                      str(model).replace('{', '').replace('}', '').replace(':', '=').replace("'", ''))
                        os.mkdir(output_dir)
                    ys_model.append(self(xs, use_model=model, model_dir=output_dir))
        if show_surr:
            ys_surr = self(xs) if from_file is None else slice_data['ys_surr']

        # Make len(qoi) by len(inputs) grid of subplots
        fig, axs = plt.subplots(len(qoi_idx), len(slice_idx), sharex='col', sharey='row')
        for i in range(len(qoi_idx)):
            for j in range(len(slice_idx)):
                if len(qoi_idx) == 1:
                    ax = axs if len(slice_idx) == 1 else axs[j]
                elif len(slice_idx) == 1:
                    ax = axs if len(qoi_idx) == 1 else axs[i]
                else:
                    ax = axs[i, j]
                x = xs[:, j, slice_idx[j]]
                if show_model is not None:
                    c = np.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1]]) if len(show_model) <= 2 else (
                        plt.get_cmap('jet')(np.linspace(0, 1, len(show_model))))
                    for k in range(len(show_model)):
                        model_str = str(show_model[k]).replace('{', '').replace('}', '').replace(':', '=').replace("'", '')
                        model_ret = ys_model[k]
                        y_model = model_ret[:, j, qoi_idx[i]]
                        label = {'best': 'High-fidelity' if len(show_model) > 1 else 'Model',
                                 'worst': 'Low-fidelity'}.get(model_str, model_str)
                        ax.plot(x, y_model, ls='-', c=c[k, :], label=label)
                if show_surr:
                    y_surr = ys_surr[:, j, qoi_idx[i]]
                    ax.plot(x, y_surr, '--r', label='Surrogate')
                ylabel = ylabels[i] if j == 0 else ''
                xlabel = xlabels[j] if i == len(qoi_idx) - 1 else ''
                legend = (i == 0 and j == len(slice_idx) - 1)
                ax_default(ax, xlabel, ylabel, legend=legend)
        fig.set_size_inches(3 * len(slice_idx), 3 * len(qoi_idx))
        fig.tight_layout()

        # Save results (unless we were already loading from a save file)
        if from_file is None and self.root_dir is not None:
            fname = f's{",".join([str(i) for i in slice_idx])}_q{",".join([str(i) for i in qoi_idx])}'
            fname = f'sweep_rand{rand_id}_' + fname if random_walk else f'sweep_nom{rand_id}_' + fname
            fdir = Path(self.root_dir) if model_dir is None else Path(model_dir) / f'sweep_{rand_id}'
            fig.savefig(fdir / f'{fname}.png', dpi=300, format='png')
            save_dict = {'slice_idx': slice_idx, 'qoi_idx': qoi_idx, 'show_model': show_model, 'show_surr': show_surr,
                         'nominal': nominal, 'random_walk': random_walk, 'xs': xs, 'ys_model': ys_model,
                         'ys_surr': ys_surr}
            with open(fdir / f'{fname}.pkl', 'wb') as fd:
                pickle.dump(save_dict, fd)

        return fig, axs

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
        # Get total cost (including offline overhead)
        train_alloc, offline_alloc, cost_cum = self.get_allocation()
        total_cost = cost_cum[-1]
        for node, alpha_dict in offline_alloc.items():
            for alpha, cost in alpha_dict.items():
                total_cost += cost[1]

        # Remove nodes with cost=0 from alloc dicts (i.e. analytical models)
        remove_nodes = []
        for node, alpha_dict in train_alloc.items():
            if len(alpha_dict) == 0:
                remove_nodes.append(node)
        for node in remove_nodes:
            del train_alloc[node]
            del offline_alloc[node]

        # Bar chart showing cost allocation breakdown for MF system at end
        fig, axs = plt.subplots(1, 2, sharey='row')
        width = 0.7
        x = np.arange(len(train_alloc))
        xlabels = list(train_alloc.keys())
        cmap = plt.get_cmap(cmap)
        for k in range(2):
            ax = axs[k]
            alloc = train_alloc if k == 0 else offline_alloc
            ax.set_title('Online training' if k == 0 else 'Overhead')
            for j, (node, alpha_dict) in enumerate(alloc.items()):
                bottom = 0
                c_intervals = np.linspace(0, 1, len(alpha_dict))
                bars = [(alpha, cost, cost[1] / total_cost) for alpha, cost in alpha_dict.items()]
                bars = sorted(bars, key=lambda ele: ele[2], reverse=True)
                for i, (alpha, cost, frac) in enumerate(bars):
                    p = ax.bar(x[j], frac, width, color=cmap(c_intervals[i]), linewidth=1,
                               edgecolor=[0, 0, 0], bottom=bottom)
                    bottom += frac
                    if frac > text_bar_width:
                        ax.bar_label(p, labels=[f'{alpha}, {round(cost[0])}'], label_type='center')
                    elif frac > arrow_bar_width:
                        xy = (x[j] + width / 2, bottom - frac / 2)  # Label smaller bars with a text off to the side
                        ax.annotate(f'{alpha}, {round(cost[0])}', xy, xytext=(xy[0] + 0.2, xy[1]),
                                    arrowprops={'arrowstyle': '->', 'linewidth': 1})
                    else:
                        pass  # Don't label really small bars
            ax_default(ax, '', "Fraction of total cost" if k == 0 else '', legend=False)
            ax.set_xticks(x, xlabels)
            ax.set_xlim(left=-1, right=x[-1] + 1)
        fig.set_size_inches(2.5*len(x), 4)
        fig.tight_layout()

        if self.root_dir is not None:
            fig.savefig(Path(self.root_dir) / 'mf_allocation.png', dpi=300, format='png')

        return fig, axs

    def get_component(self, comp_name: str) -> ComponentSurrogate:
        """Return the `ComponentSurrogate` object for this component.

        :param comp_name: name of the component to return
        :returns: the `ComponentSurrogate` object
        """
        return self.graph.nodes[comp_name]['surrogate']

    def _print_title_str(self, title_str: str):
        """Log an important message."""
        self.logger.info('-' * int(len(title_str)/2) + title_str + '-' * int(len(title_str)/2))

    def save_to_file(self, filename: str, save_dir: str | Path = None):
        """Save the SystemSurrogate object to a .pkl file.

        :param filename: filename of the .pkl file to save to
        :param save_dir: overrides existing surrogate root directory if provided
        """
        if self.root_dir is None and save_dir is None:
            # Can't save to file if root_dir is None
            return

        save_dir = save_dir if save_dir is not None else str(Path(self.root_dir) / 'sys')
        if not Path(save_dir).is_dir():
            save_dir = '.'

        exec_temp = self.executor   # Temporarily save executor obj (can't pickle it)
        self.set_executor(None)
        with open(Path(save_dir) / filename, 'wb') as dill_file:
            dill.dump(self, dill_file)
        self.set_executor(exec_temp)
        self.logger.info(f'SystemSurrogate saved to {(Path(save_dir) / filename).resolve()}')

    def _set_output_dir(self, set_dict: dict[str: str | Path]):
        """Set the output directory for each component in `set_dict`.

        :param set_dict: a `dict` of component names (`str`) to their new output directories
        """
        for node, node_obj in self.graph.nodes.items():
            if node in set_dict:
                node_obj['surrogate']._set_output_dir(set_dict.get(node))

    def set_root_directory(self, root_dir: str | Path, stdout: bool = True):
        """Set the root to a new directory, for example if you move to a new filesystem.

        :param root_dir: new root directory
        :param stdout: whether to connect the logger to console (default)
        """
        self.root_dir = str(Path(root_dir).resolve())
        log_file = None
        if not (Path(self.root_dir) / 'sys').is_dir():
            os.mkdir(Path(self.root_dir) / 'sys')
        if not (Path(self.root_dir) / 'components').is_dir():
            os.mkdir(Path(self.root_dir) / 'components')
        for f in os.listdir(self.root_dir):
            if f.endswith('.log'):
                log_file = str((Path(self.root_dir) / f).resolve())
                break
        if log_file is None:
            fname = datetime.datetime.now(tz=timezone.utc).isoformat().split('.')[0].replace(':', '.') + 'UTC_sys.log'
            log_file = str((Path(self.root_dir) / fname).resolve())

        # Setup the log file
        self.log_file = log_file
        self.logger = get_logger(self.__class__.__name__, log_file=log_file, stdout=stdout)

        # Update model output directories
        for node, node_obj in self.graph.nodes.items():
            surr = node_obj['surrogate']
            surr.logger = get_logger(surr.__class__.__name__, log_file=log_file, stdout=stdout)
            surr.log_file = self.log_file
            if surr.save_enabled():
                output_dir = str((Path(self.root_dir) / 'components' / node).resolve())
                if not Path(output_dir).is_dir():
                    os.mkdir(output_dir)
                surr._set_output_dir(output_dir)

    def __getitem__(self, component: str) -> ComponentSurrogate:
        """Convenience method to get the `ComponentSurrogate object` from the `SystemSurrogate`.

        :param component: the name of the component to get
        :returns: the `ComponentSurrogate` object
        """
        return self.get_component(component)

    def __repr__(self):
        s = f'----SystemSurrogate----\nAdjacency: \n{nx.to_numpy_array(self.graph, dtype=int)}\n' \
            f'Exogenous inputs: {[str(var) for var in self.exo_vars]}\n'
        for node, node_obj in self.graph.nodes.items():
            s += f'Component: {node}\n{node_obj["surrogate"]}'
        return s

    def __str__(self):
        return self.__repr__()

    def set_executor(self, executor: Executor | None):
        """Set a new `concurrent.futures.Executor` object for parallel calls.

        :param executor: the new `Executor` object
        """
        self.executor = executor
        for node, node_obj in self.graph.nodes.items():
            node_obj['surrogate'].executor = executor

    @staticmethod
    def load_from_file(filename: str | Path, root_dir: str | Path = None, executor: Executor = None):
        """Load a `SystemSurrogate object` from file.

        :param filename: the .pkl file to load
        :param root_dir: folder to use as the root directory, (uses file's second parent directory by default)
        :param executor: a `concurrent.futures.Executor` object to set; clears it if None
        :returns: the `SystemSurrogate` object
        """
        if root_dir is None:
            root_dir = Path(filename).parent.parent     # Assume root/sys/filename.pkl

        with open(Path(filename), 'rb') as dill_file:
            sys_surr = dill.load(dill_file)
            sys_surr.set_executor(executor)
            sys_surr.set_root_directory(root_dir)
            sys_surr.logger.info(f'SystemSurrogate loaded from {Path(filename).resolve()}')

        return sys_surr

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
