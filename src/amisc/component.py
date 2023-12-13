"""`component.py`

A Component is an `amisc` wrapper around a single discipline model. It manages surrogate construction and optionally
a hierarchy of modeling fidelities that may be available. Concrete component classes all inherit from the base
`ComponentSurrogate` class provided here. Components manage an array of `BaseInterpolator` objects to form a
multifidelity hierarchy.

Includes
--------
- `ComponentSurrogate`: the base class that is fundamental to the adaptive multi-index stochastic collocation strategy
- `SparseGridSurrogate`: an AMISC component that manages a hierarchy of `LagrangeInterpolator` objects
- `AnalyticalSurrogate`: a light wrapper around a single discipline model that does not require surrogate approximation
"""
import itertools
import copy
import ast
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import Executor, ALL_COMPLETED, wait

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from amisc.utils import get_logger
from amisc.rv import BaseRV
from amisc import IndexSet, MiscTree, InterpResults
from amisc.interpolator import BaseInterpolator, LagrangeInterpolator


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
    :vartype x_vars: list[BaseRV]
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

    def __init__(self, x_vars: list[BaseRV] | BaseRV, model: callable,
                 multi_index: IndexSet = None,
                 truth_alpha: tuple = (), max_alpha: tuple = (), max_beta: tuple = (),
                 log_file: str | Path = None, executor: Executor = None,
                 model_args: tuple = (), model_kwargs: dict = None):
        """Construct the MISC surrogate and initialize with any multi-indices passed in.

        !!! Info "Model specification"
            The model is a callable function of the form `ret = model(x, *args, **kwargs)`. The return value is a
            dictionary of the form `ret = {'y': y, 'files': files, 'cost': cost}`. In the return dictionary, you
            specify the raw model output `y` as an `np.ndarray` at a _minimum_. Optionally, you can specify paths to
            output files and the average model cost (in units of seconds of cpu time), and anything else you want.

        !!! Warning
            If the model has multiple fidelities, then the function signature must be `model(x, alpha, *args, **kwargs)`
            ; the first argument after `x` will always be the fidelity indices `alpha`. The rest of `model_args` will
            be passed in after (you do not need to include `alpha` in `model_args`, it is done automatically).

        :param x_vars: `[X1, X2, ...]` list of variables specifying bounds/pdfs for each input
        :param model: the function to approximate, callable as `ret = model(x, *args, **kwargs)`
        :param multi_index: `[((alpha1), (beta1)), ... ]` list of concatenated multi-indices $(\\alpha, \\beta)$
        :param truth_alpha: specifies the highest model fidelity indices necessary for a "ground truth" comparison
        :param max_alpha: the maximum model refinement indices to allow, defaults to `(2,...)` if applicable
        :param max_beta: the maximum surrogate refinement indices, defaults to `(2,...)` of length `x_dim`
        :param log_file: specifies a log file (optional)
        :param executor: parallel executor used to add candidate indices in parallel (optional)
        :param model_args: optional args to pass when calling the model
        :param model_kwargs: optional kwargs to pass when calling the model
        """
        self.logger = get_logger(self.__class__.__name__, log_file=log_file)
        self.log_file = log_file
        self.executor = executor
        self.training_flag = None  # Keep track of which MISC coeffs are active
        # (True=active set, False=active+candidate sets, None=Neither/unknown)

        multi_index = list() if multi_index is None else multi_index
        assert self.is_downward_closed(multi_index), 'Must be a downward closed set.'
        self.ydim = None
        self.index_set = []         # The active index set for the MISC approximation
        self.candidate_set = []     # Candidate indices for refinement
        self._model = model
        self._model_args = model_args
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.truth_alpha = truth_alpha
        self.x_vars = x_vars if isinstance(x_vars, list) else [x_vars]
        max_alpha = (2,)*len(truth_alpha) if max_alpha == () else max_alpha
        max_beta = (2,)*len(self.x_vars) if max_beta == () else max_beta
        self.max_refine = list(max_alpha + max_beta)    # Max refinement indices

        # Initialize important tree-like structures
        self.surrogates = dict()        # Maps alphas -> betas -> surrogates
        self.costs = dict()             # Maps alphas -> betas -> wall clock run times
        self.misc_coeff = dict()        # Maps alphas -> betas -> MISC coefficients

        # Construct vectors of [0,1]^dim(alpha+beta)
        Nij = len(self.max_refine)
        self.ij = np.zeros((2 ** Nij, Nij), dtype=np.uint8)
        for i, ele in enumerate(itertools.product([0, 1], repeat=Nij)):
            self.ij[i, :] = ele

        # Initialize any indices that were passed in
        multi_index = list() if multi_index is None else multi_index
        for alpha, beta in multi_index:
            self.activate_index(alpha, beta)

    def activate_index(self, alpha: tuple, beta: tuple):
        """Add a multi-index to the active set and all neighbors to the candidate set.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        """
        # User is responsible for making sure index set is downward-closed
        self.add_surrogate(alpha, beta)
        ele = (alpha, beta)
        if ele in self.index_set:
            self.logger.warning(f'Multi-index {ele} is already in the active index set. Ignoring...')
            return

        # Add all possible new candidates (distance of one unit vector away)
        ind = list(alpha + beta)
        new_candidates = []
        for i in range(len(ind)):
            ind_new = ind.copy()
            ind_new[i] += 1

            # Don't add if we surpass a refinement limit
            if np.any(np.array(ind_new) > np.array(self.max_refine)):
                continue

            # Add the new index if it maintains downward-closedness
            new_cand = (tuple(ind_new[:len(alpha)]), tuple(ind_new[len(alpha):]))
            down_closed = True
            for j in range(len(ind)):
                ind_check = ind_new.copy()
                ind_check[j] -= 1
                if ind_check[j] >= 0:
                    tup_check = (tuple(ind_check[:len(alpha)]), tuple(ind_check[len(alpha):]))
                    if tup_check not in self.index_set and tup_check != ele:
                        down_closed = False
                        break
            if down_closed:
                new_candidates.append(new_cand)

        # Build an interpolator for each new candidate
        if self.executor is None:   # Sequential
            for a, b in new_candidates:
                self.add_surrogate(a, b)
        else:                       # Parallel
            temp_exc = self.executor
            self.executor = None
            for a, b in new_candidates:
                if str(a) not in self.surrogates:
                    self.surrogates[str(a)] = dict()
                    self.costs[str(a)] = dict()
                    self.misc_coeff[str(a)] = dict()
            self.parallel_add_candidates(new_candidates, temp_exc)
            self.executor = temp_exc

        # Move to the active index set
        if ele in self.candidate_set:
            self.candidate_set.remove(ele)
        self.index_set.append(ele)
        new_candidates = [cand for cand in new_candidates if cand not in self.candidate_set]
        self.candidate_set.extend(new_candidates)
        self.training_flag = None   # Makes sure misc coeffs get recomputed next time

    def add_surrogate(self, alpha: tuple, beta: tuple):
        """Build a `BaseInterpolator` object for a given $(\\alpha, \\beta)$

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        """
        # Create a dictionary for each alpha model to store multiple surrogate fidelities (beta)
        if str(alpha) not in self.surrogates:
            self.surrogates[str(alpha)] = dict()
            self.costs[str(alpha)] = dict()
            self.misc_coeff[str(alpha)] = dict()

        # Create a new interpolator object for this multi-index (abstract method)
        if self.surrogates[str(alpha)].get(str(beta), None) is None:
            self.logger.info(f'Building interpolator for index {(alpha, beta)} ...')
            x_new_idx, x_new, interp = self.build_interpolator(alpha, beta)
            self.surrogates[str(alpha)][str(beta)] = interp
            cost = self.update_interpolator(x_new_idx, x_new, interp)  # Awkward, but needed to separate the model evals
            self.costs[str(alpha)][str(beta)] = cost
            if self.ydim is None:
                self.ydim = interp.ydim()

    def init_coarse(self):
        """Initialize the coarsest interpolation and add to the active index set"""
        alpha = (0,) * len(self.truth_alpha)
        beta = (0,) * len(self.max_refine[len(self.truth_alpha):])
        self.activate_index(alpha, beta)

    def iterate_candidates(self):
        """Iterate candidate indices one by one into the active index set.

        :yields alpha, beta: the multi-indices of the current candidate that has been moved to active set
        """
        for alpha, beta in list(self.candidate_set):
            # Temporarily add a candidate index to active set
            self.index_set.append((alpha, beta))
            yield alpha, beta
            del self.index_set[-1]

    def predict(self, x: np.ndarray | float, use_model: str | tuple = None, model_dir: str | Path = None,
                training: bool = False, index_set: IndexSet = None) -> np.ndarray:
        """Evaluate the MISC approximation at new points `x`.

        !!! Note
            By default this will predict the MISC surrogate approximation. However, for convenience you can also specify
            `use_model` to call the underlying function instead.

        :param x: `(..., x_dim)` the points to be interpolated, must be within input domain for accuracy
        :param use_model: 'best'=high-fidelity, 'worst'=low-fidelity, tuple=a specific `alpha`, None=surrogate (default)
        :param model_dir: directory to save output files if `use_model` is specified, ignored otherwise
        :param training: if `True`, then only compute with the active index set, otherwise use all candidates as well
        :param index_set: a list of concatenated $(\\alpha, \\beta)$ to override `self.index_set` if given, else ignore
        :returns y: `(..., y_dim)` the surrogate approximation of the function (or the function itself if `use_model`)
        """
        if use_model is not None:
            return self._bypass_surrogate(x, use_model, model_dir)

        index_set, misc_coeff = self._combination(index_set, training)  # Choose the correct index set and misc_coeff

        y = np.zeros(x.shape[:-1] + (self.ydim,))
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                func = self.surrogates[str(alpha)][str(beta)]
                y += comb_coeff * func(x)

        return y

    def __call__(self, *args, **kwargs):
        """Here for convenience so you can also do `ret = surrogate(x)`, just like the `BaseInterpolator`."""
        return self.predict(*args, **kwargs)

    def update_misc_coeffs(self, index_set: IndexSet = None) -> MiscTree:
        """Update the combination technique coeffs for MISC using the given index set.

        :param index_set: the index set to consider when computing the MISC coefficients, defaults to the active set
        :returns: the MISC coefficients for the given index set ($\\alpha$ -> $\\beta$ -> coeff)
        """
        if index_set is None:
            index_set = self.index_set

        # Construct a (N_indices, dim(alpha+beta)) refactor of the index_set for arrayed computations
        index_mat = np.zeros((len(index_set), len(self.max_refine)), dtype=np.uint8)
        for i, (alpha, beta) in enumerate(index_set):
            index_mat[i, :] = alpha + beta
        index_mat = np.expand_dims(index_mat, axis=0)                               # (1, Ns, Nij)

        misc_coeff = dict()
        for alpha, beta in index_set:
            # Add permutations of [0, 1] to (alpha, beta)
            alpha_beta = np.array(alpha+beta, dtype=np.uint8)[np.newaxis, :]        # (1, Nij)
            new_indices = np.expand_dims(alpha_beta + self.ij, axis=1)              # (2**Nij, 1, Nij)

            # Find which indices are in the index_set (using np broadcasting comparison)
            diff = new_indices - index_mat                                  # (2**Nij, Ns, Nij)
            idx = np.count_nonzero(diff, axis=-1) == 0                      # (2**Nij, Ns)
            idx = np.any(idx, axis=-1)                                      # (2**Nij,)
            ij_use = self.ij[idx, :]                                        # (*, Nij)
            l1_norm = np.sum(np.abs(ij_use), axis=-1)                       # (*,)
            coeff = np.sum((-1) ** l1_norm)                                 # float

            # Save misc coeff to a dict() tree structure
            if misc_coeff.get(str(alpha)) is None:
                misc_coeff[str(alpha)] = dict()
            misc_coeff[str(alpha)][str(beta)] = coeff
            self.misc_coeff[str(alpha)][str(beta)] = coeff

        return misc_coeff

    def get_sub_surrogate(self, alpha: tuple, beta: tuple) -> BaseInterpolator:
        """Get the specific sub-surrogate corresponding to the $(\\alpha, \\beta)$ fidelity.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        :returns: the corresponding `BaseInterpolator` object
        """
        return self.surrogates[str(alpha)][str(beta)]

    def get_cost(self, alpha: tuple, beta: tuple) -> float:
        """Return the total cost (wall time s) required to add $(\\alpha, \\beta)$ to the MISC approximation.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        """
        try:
            return self.costs[str(alpha)][str(beta)]
        except:
            return 0.0

    def update_input_bds(self, idx: int, bds: tuple):
        """Update the bounds of the input variable at the given index.

        :param idx: the index of the input variable to update
        :param bds: the new bounds
        """
        self.x_vars[int(idx)].update_bounds(*bds)

        # Update the bounds in all associated surrogates
        for alpha in self.surrogates:
            for beta in self.surrogates[alpha]:
                self.surrogates[alpha][beta].update_input_bds(idx, bds)

    def save_enabled(self):
        """Return whether this model wants to save outputs to file.

        !!! Note
            You can specify that a model wants to save outputs to file by providing an `'output_dir'` kwarg.
        """
        return self._model_kwargs.get('output_dir') is not None

    def _set_output_dir(self, output_dir: str | Path):
        """Update the component model output directory.

        :param output_dir: the new directory for model output files
        """
        if output_dir is not None:
            output_dir = str(Path(output_dir).resolve())
        self._model_kwargs['output_dir'] = output_dir
        for alpha in self.surrogates:
            for beta in self.surrogates[alpha]:
                self.surrogates[alpha][beta]._model_kwargs['output_dir'] = output_dir

    def __repr__(self):
        """Shows all multi-indices in the current approximation and their corresponding MISC coefficients."""
        s = f'Inputs \u2014 {[str(var) for var in self.x_vars]}\n'
        if self.training_flag is None:
            self.update_misc_coeffs()
            self.training_flag = True

        if self.training_flag:
            s += '(Training mode)\n'
            for alpha, beta in self.index_set:
                s += f"[{int(self.misc_coeff[str(alpha)][str(beta)])}] \u2014 {alpha}, {beta}\n"
            for alpha, beta in self.candidate_set:
                s += f"[-] \u2014 {alpha}, {beta}\n"
        else:
            s += '(Evaluation mode)\n'
            for alpha, beta in self.index_set + self.candidate_set:
                s += f"[{int(self.misc_coeff[str(alpha)][str(beta)])}] \u2014 {alpha}, {beta}\n"
        return s

    def __str__(self):
        """Everyone will view these objects the same way."""
        return self.__repr__()

    def _bypass_surrogate(self, x, use_model, model_dir):
        """Bypass surrogate evaluation and use the specified model"""
        output_dir = self._model_kwargs.get('output_dir')
        if self.save_enabled():
            self._model_kwargs['output_dir'] = model_dir

        alpha_use = {'best': self.truth_alpha, 'worst': (0,) * len(self.truth_alpha)}.get(use_model, use_model)
        kwargs = copy.deepcopy(self._model_kwargs)
        if len(alpha_use) > 0:
            kwargs['alpha'] = alpha_use
        ret = self._model(x, *self._model_args, **kwargs)

        if output_dir is not None:
            self._model_kwargs['output_dir'] = output_dir

        if not isinstance(ret, dict):
            self.logger.warning(f"Function {self._model} did not return a dict of the form {{'y': y}}. Please make sure"
                                f" you do so to avoid conflicts. Returning the value directly instead...")

        return ret['y'] if isinstance(ret, dict) else ret

    def _combination(self, index_set, training):
        """Decide which index set and corresponding misc coefficients to use."""
        misc_coeff = copy.deepcopy(self.misc_coeff)
        if index_set is None:
            # Use active indices + candidate indices depending on training mode
            index_set = self.index_set if training else self.index_set + self.candidate_set

            # Decide when to update misc coefficients
            if self.training_flag is None:
                misc_coeff = self.update_misc_coeffs(index_set)  # On initialization or reset
            else:
                if (not self.training_flag and training) or (self.training_flag and not training):
                    misc_coeff = self.update_misc_coeffs(index_set)  # Logical XOR cases for training mode

            # Save an indication of what state the MISC coefficients are in (i.e. training or eval mode)
            self.training_flag = training
        else:
            # If we passed in an index set, always recompute misc coeff and toggle for reset on next call
            misc_coeff = self.update_misc_coeffs(index_set)
            self.training_flag = None

        return index_set, misc_coeff

    @staticmethod
    def is_one_level_refinement(beta_old: tuple, beta_new: tuple) -> bool:
        """Check if a new `beta` multi-index is a one-level refinement from a previous `beta`.

        !!! Example
            Refining from `(0, 1, 2)` to the new multi-index `(1, 1, 2)` is a one-level refinement. But refining to
            either `(2, 1, 2)` or `(1, 2, 2)` are not, since more than one refinement occurs at the same time.

        :param beta_old: the starting multi-index
        :param beta_new: the new refined multi-index
        :returns: whether `beta_new` is a one-level refinement from `beta_old`
        """
        level_diff = np.array(beta_new, dtype=int) - np.array(beta_old, dtype=int)
        ind = np.nonzero(level_diff)[0]
        return ind.shape[0] == 1 and level_diff[ind] == 1

    @staticmethod
    def is_downward_closed(indices: IndexSet) -> bool:
        """Return if a list of $(\\alpha, \\beta)$ multi-indices is downward-closed.

        MISC approximations require a downward-closed set in order to use the combination-technique formula for the
        coefficients (as implemented here).

        !!! Example
            The list `[( (0,), (0,) ), ( (1,), (0,) ), ( (1,), (1,) )]` is downward-closed. You can visualize this as
            building a stack of cubes: in order to place a cube, all adjacent cubes must be present (does the logo
            make sense now?).

        :param indices: list() of (`alpha`, `beta`) multi-indices
        :returns: whether the set of indices is downward-closed
        """
        # Iterate over every multi-index
        for alpha, beta in indices:
            # Every smaller multi-index must also be included in the indices list
            sub_sets = [np.arange(tuple(alpha+beta)[i]+1) for i in range(len(alpha) + len(beta))]
            for ele in itertools.product(*sub_sets):
                tup = (tuple(ele[:len(alpha)]), tuple(ele[len(alpha):]))
                if tup not in indices:
                    return False
        return True

    @abstractmethod
    def build_interpolator(self, alpha: tuple, beta: tuple) -> InterpResults:
        """Return a `BaseInterpolator` object and new refinement points for a given $(\\alpha, \\beta)$ multi-index.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        :returns: `idx`, `x`, `interp` - list of new grid indices, the new grid points `(N_new, x_dim)`, and the
                  `BaseInterpolator` object. Similar to `BaseInterpolator.refine()`.
        """
        pass

    @abstractmethod
    def update_interpolator(self, x_new_idx: list[int | tuple | str], x_new: np.ndarray, interp: BaseInterpolator) -> float:
        """Secondary method to actually compute and save model evaluations within the interpolator.

        !!! Note
            This distinction with `build_interpolator` was necessary to separately construct the interpolator and be
            able to evaluate the model at the new interpolation points. You can see that `parallel_add_candidates`
            uses this distinction to compute the model in parallel on MPI workers, for example.

        :param x_new_idx: list of new grid point indices
        :param x_new: `(N_new, x_dim)`, the new grid point locations
        :param interp: the `BaseInterpolator` object to compute model evaluations with
        :returns cost: the cost (in wall time seconds) required to add this `BaseInterpolator` object
        """
        pass

    @abstractmethod
    def parallel_add_candidates(self, candidates: IndexSet, executor: Executor):
        """Defines a function to handle adding candidate indices in parallel.

        !!! Note
            While `build_interpolator` can make changes to 'self', these changes will not be saved in the master task
            if running in parallel over MPI workers, for example. This method is a workaround so that all required
            mutable changes to 'self' are made in the master task, before distributing tasks to parallel workers
            using this method. You can pass if you don't plan to add candidates in parallel.

        :param candidates: list of [(alpha, beta),...] multi-indices
        :param executor: the executor used to iterate candidates in parallel
        """
        pass


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
        self.curr_max_beta = dict()     # Maps alphas -> current max refinement indices
        self.x_grids = dict()           # Maps alphas -> list of ndarrays specifying 1d grids corresponding to max_beta
        self.xi_map = dict()            # Maps alphas -> grid point coords -> interpolation points
        self.yi_map = dict()            # Maps alphas -> grid point coords -> interpolation function values
        self.yi_nan_map = dict()        # Maps alphas -> grid point coords -> interpolated yi values when yi=nan
        self.yi_files = dict()          # Maps alphas -> grid point coords -> model output files (optional)
        super().__init__(*args, **kwargs)

    # Override
    def predict(self, x, use_model=None, model_dir=None, training=False, index_set=None):
        """Need to override `super()` to allow passing in interpolation grids `xi` and `yi`."""
        if use_model is not None:
            return self._bypass_surrogate(x, use_model, model_dir)

        index_set, misc_coeff = self._combination(index_set, training)

        y = np.zeros(x.shape[:-1] + (self.ydim,))
        for alpha, beta in index_set:
            comb_coeff = misc_coeff[str(alpha)][str(beta)]
            if np.abs(comb_coeff) > 0:
                # Gather the xi/yi interpolation points/qoi_ind for this sub tensor-product grid
                interp = self.surrogates[str(alpha)][str(beta)]
                xi, yi = self.get_tensor_grid(alpha, beta)

                # Add this sub tensor-product grid to the MISC approximation
                y += comb_coeff * interp(x, xi=xi, yi=yi)

        return y

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
        coords = [np.arange(grid_sizes[n]) for n in range(interp.xdim())]
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
                y_interp[~nan_idx] = yi[~nan_idx]   # Only keep imputed values where yi is nan
                self.yi_nan_map[str(alpha)][str(grid_coord)] = y_interp

        # Go back and try to re-interpolate old nan values as more points get added to the grid
        if imputer is not None:
            for grid_coord in list(self.yi_nan_map[str(alpha)].keys()):
                if grid_coord not in yi_dict:
                    x_interp = self.xi_map[str(alpha)][str(grid_coord)].reshape((1, xdim))
                    y_interp = imputer.predict(x_interp)
                    self.yi_nan_map[str(alpha)][str(grid_coord)] = np.atleast_1d(np.squeeze(y_interp))

    # Override
    def get_sub_surrogate(self, alpha: tuple, beta: tuple, include_grid: bool = False) -> BaseInterpolator:
        """Get the specific sub-surrogate corresponding to the $(\\alpha, \\beta)$ fidelity.

        :param alpha: A multi-index specifying model fidelity
        :param beta: A multi-index specifying surrogate fidelity
        :param include_grid: whether to add the `xi/yi` interpolation points to the returned `BaseInterpolator` object
        :returns: the `BaseInterpolator` object corresponding to $(\\alpha, \\beta)$
        """
        interp = super().get_sub_surrogate(alpha, beta)
        if include_grid:
            interp.xi, interp.yi = self.get_tensor_grid(alpha, beta)
        return interp

    def build_interpolator(self, alpha, beta):
        """Abstract method implementation for constructing the tensor-product grid interpolator."""
        # Create a new tensor-product grid interpolator for the base index (0, 0, ...)
        if np.sum(beta) == 0:
            kwargs = copy.deepcopy(self._model_kwargs)
            if len(alpha) > 0:
                kwargs['alpha'] = alpha
            interp = LagrangeInterpolator(beta, self.x_vars, model=self._model, model_args=self._model_args,
                                          model_kwargs=kwargs, init_grids=True, reduced=True)
            x_pt = np.array([float(interp.x_grids[n][beta[n]]) for n in range(interp.xdim())], dtype=np.float32)
            self.curr_max_beta[str(alpha)] = list(beta)
            self.x_grids[str(alpha)] = copy.deepcopy(interp.x_grids)
            self.xi_map[str(alpha)] = {str(beta): x_pt}
            self.yi_map[str(alpha)] = dict()
            self.yi_nan_map[str(alpha)] = dict()
            if self.save_enabled():
                self.yi_files[str(alpha)] = dict()

            return [beta], x_pt.reshape((1, len(self.x_vars))), interp
        # Otherwise, all other indices are a refinement of previous grids

        # Look for first multi-index neighbor that is one level of refinement away
        refine_tup = None
        for beta_old_str in list(self.surrogates[str(alpha)].keys()):
            beta_old = ast.literal_eval(beta_old_str)
            if self.is_one_level_refinement(beta_old, beta):
                idx_refine = int(np.nonzero(np.array(beta, dtype=int) - np.array(beta_old, dtype=int))[0])
                refine_level = beta[idx_refine]
                if refine_level > self.curr_max_beta[str(alpha)][idx_refine]:
                    # Generate next refinement grid and save (refine_tup = tuple(x_new_idx, x_new, interp))
                    refine_tup = self.surrogates[str(alpha)][beta_old_str].refine(beta, auto=False)
                    self.curr_max_beta[str(alpha)][idx_refine] = refine_level
                    self.x_grids[str(alpha)][idx_refine] = copy.deepcopy(refine_tup[2].x_grids[idx_refine])
                else:
                    # Access the refinement grid from memory (it is already computed)
                    num_pts = self.surrogates[str(alpha)][beta_old_str].get_grid_sizes(beta)[idx_refine]
                    x_refine = self.x_grids[str(alpha)][idx_refine][:num_pts]
                    refine_tup = self.surrogates[str(alpha)][beta_old_str].refine(beta, x_refine=x_refine,
                                                                                  auto=False)
                break  # Only need to grab one neighbor

        # Gather new interpolation grid points
        x_new_idx, x_new, interp = refine_tup
        xn_coord = []   # Save multi-index coordinates of points to compute model at for refinement
        xn_pts = np.zeros((0, interp.xdim()), dtype=np.float32)     # Save physical x location of new points
        for i, multi_idx in enumerate(x_new_idx):
            if str(multi_idx) not in self.yi_map[str(alpha)]:
                # We have not computed this grid coordinate yet
                xn_coord.append(multi_idx)
                xn_pts = np.concatenate((xn_pts, x_new[i, np.newaxis, :]), axis=0)  # (N_new, xdim)
                self.xi_map[str(alpha)][str(multi_idx)] = x_new[i, :]

        return xn_coord, xn_pts, interp

    def update_interpolator(self, x_new_idx, x_new, interp):
        """Awkward solution, I know, but actually compute and save the model evaluations here."""
        # Compute and store model output at new refinement points in a hash structure
        yi_ret = interp.set_yi(x_new=(x_new_idx, x_new))

        if self.ydim is None:
            for coord_str, yi in yi_ret['y'].items():
                self.ydim = yi.shape[0]
                break

        alpha = interp._model_kwargs.get('alpha', ())
        self.update_yi(alpha, interp.beta, yi_ret['y'])
        if self.save_enabled():
            self.yi_files[str(alpha)].update(yi_ret['files'])
        cost = interp.model_cost * len(x_new_idx)

        return cost

    def parallel_add_candidates(self, candidates: IndexSet, executor: Executor):
        """Work-around to make sure mutable instance variable changes are made before/after
        splitting tasks using this method over parallel (potentially MPI) workers. You can pass if you are not
        interested in such parallel ideas.

        !!! Warning
            MPI workers cannot save changes to `self` so this method should only distribute static tasks to the workers.

        :param candidates: list of [(alpha, beta),...] multi-indices
        :param executor: the executor used to iterate candidates in parallel
        """
        # Do sequential tasks first (i.e. make mutable changes to self), build up parallel task args
        task_args = []
        for alpha, beta in candidates:
            x_new_idx, x_new, interp = self.build_interpolator(alpha, beta)
            task_args.append((alpha, beta, x_new_idx, x_new, interp))

        def parallel_task(alpha, beta, x_new_idx, x_new, interp):
            # Must return anything you want changed in self or interp (mutable changes aren't saved over MPI workers)
            logger = get_logger(self.__class__.__name__, log_file=self.log_file, stdout=False)
            logger.info(f'Building interpolator for index {(alpha, beta)} ...')
            yi_ret = interp.set_yi(x_new=(x_new_idx, x_new))
            model_cost = interp.model_cost if interp.model_cost is not None else 1
            return yi_ret, model_cost

        # Wait for all parallel workers to return
        fs = [executor.submit(parallel_task, *args) for args in task_args]
        wait(fs, timeout=None, return_when=ALL_COMPLETED)

        # Update self and interp with the results from all workers (and check for errors)
        for i, future in enumerate(fs):
            try:
                a = task_args[i][0]
                b = task_args[i][1]
                x_new_idx = task_args[i][2]
                interp = task_args[i][4]
                yi_ret, model_cost = future.result()
                interp.model_cost = model_cost
                self.surrogates[str(a)][str(b)] = interp
                self.update_yi(a, b, yi_ret['y'])
                if self.save_enabled():
                    self.yi_files[str(a)].update(yi_ret['files'])
                self.costs[str(a)][str(b)] = interp.model_cost * len(x_new_idx)

                if self.ydim is None:
                    for coord_str, yi in self.yi_map[str(a)].items():
                        self.ydim = yi.shape[0]
                        break
            except:
                self.logger.error(f'An exception occurred in a thread handling build_interpolator{candidates[i]}')
                raise


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
