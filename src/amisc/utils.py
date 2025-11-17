"""Provides some basic utilities for the package.

Includes:

- `to_model_dataset` — convert surrogate input/output dataset to a form usable by the true model
- `to_surrogate_dataset` — convert true model input/output dataset to a form usable by the surrogate
- `constrained_lls` — solve a constrained linear least squares problem
- `search_for_file` — search for a file in the current working directory and additional search paths
- `format_inputs` — broadcast and reshape all inputs to the same shape
- `format_outputs` — reshape all outputs to a common loop shape
- `parse_function_string` — convert function-like strings to arguments and keyword-arguments
- `relative_error` — compute the relative L2 error between two vectors
- `get_logger` — logging utility with nice formatting
"""
from __future__ import annotations

import ast
import copy
import inspect
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

__all__ = ['parse_function_string', 'relative_error', 'get_logger', 'format_inputs', 'format_outputs',
           'search_for_file', 'constrained_lls', 'to_surrogate_dataset', 'to_model_dataset']

from amisc.typing import COORDS_STR_ID, LATENT_STR_ID, Dataset

if TYPE_CHECKING:
    import amisc.variable

LOG_FORMATTER = logging.Formatter(u"%(asctime)s — [%(levelname)s] — %(name)-15s — %(message)s")


def _combine_latent_arrays(arr):
    """Helper function to concatenate latent arrays into a single variable in the `arr` Dataset."""
    for var in list(arr.keys()):
        if LATENT_STR_ID in var:  # extract latent variables from surrogate data
            base_id = str(var).split(LATENT_STR_ID)[0]
            arr[base_id] = arr[var][..., np.newaxis] if arr.get(base_id) is None else (
                np.concatenate((arr[base_id], arr[var][..., np.newaxis]), axis=-1))
            del arr[var]


def to_surrogate_dataset(dataset: Dataset, variables: 'amisc.variable.VariableList', del_fields: bool = True,
                         **field_coords) -> tuple[Dataset, list[str]]:
    """Convert true model input/output dataset to a form usable by the surrogate. Primarily, compress field
    quantities and normalize.

    :param dataset: the dataset to convert
    :param variables: the `VariableList` containing the variable objects used in `dataset` -- these objects define
                      the normalization and compression methods to use for each variable
    :param del_fields: whether to delete the original field quantities from the dataset after compression
    :param field_coords: pass in extra field qty coords as f'{var}_coords' for compression (optional)
    :returns: the compressed/normalized dataset and a list of variable names to pass to surrogate
    """
    surr_vars = []
    dataset = copy.deepcopy(dataset)
    for var in variables:
        # Only grab scalars in the dataset or field qtys if all fields are present
        if var in dataset or (var.compression is not None and all([f in dataset for f in var.compression.fields])):
            if var.compression is not None:
                coords = dataset.get(f'{var}{COORDS_STR_ID}', field_coords.get(f'{var}{COORDS_STR_ID}', None))
                latent = var.compress({field: dataset[field] for field in
                                       var.compression.fields}, coords=coords)['latent']  # all fields must be present
                for i in range(latent.shape[-1]):
                    dataset[f'{var.name}{LATENT_STR_ID}{i}'] = latent[..., i]
                    surr_vars.append(f'{var.name}{LATENT_STR_ID}{i}')
                if del_fields:
                    for field in var.compression.fields:
                        del dataset[field]
                    if dataset.get(f'{var}{COORDS_STR_ID}', None) is not None:
                        del dataset[f'{var}{COORDS_STR_ID}']
            else:
                dataset[var.name] = var.normalize(dataset[var.name])
                surr_vars.append(f'{var.name}')

    return dataset, surr_vars


def to_model_dataset(dataset: Dataset, variables: 'amisc.variable.VariableList', del_latent: bool = True,
                     **field_coords) -> tuple[Dataset, Dataset]:
    """Convert surrogate input/output dataset to a form usable by the true model. Primarily, reconstruct
    field quantities and denormalize.

    :param dataset: the dataset to convert
    :param variables: the `VariableList` containing the variable objects used in `dataset` -- these objects define
                      the normalization and compression methods to use for each variable
    :param del_latent: whether to delete the latent variables from the dataset after reconstruction
    :param field_coords: pass in extra field qty coords as f'{var}_coords' for reconstruction (optional)
    :returns: the reconstructed/denormalized dataset and any field coordinates used during reconstruction
    """
    dataset = copy.deepcopy(dataset)
    _combine_latent_arrays(dataset)

    ret_coords = {}
    for var in variables:
        if var in dataset:
            if var.compression is not None:
                # coords = self.model_kwargs.get(f'{var.name}_coords', None)
                coords = field_coords.get(f'{var}{COORDS_STR_ID}', None)
                field = var.reconstruct({'latent': dataset[var]}, coords=coords)
                if del_latent:
                    del dataset[var]
                coords = field.pop('coords')
                ret_coords[f'{var.name}{COORDS_STR_ID}'] = copy.deepcopy(coords)
                dataset.update(field)
            else:
                dataset[var] = var.denormalize(dataset[var])

    return dataset, ret_coords


def constrained_lls(A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Minimize $||Ax-b||_2$, subject to $Cx=d$, i.e. constrained linear least squares.

    !!! Note
        See [these lecture notes](http://www.seas.ucla.edu/~vandenbe/133A/lectures/cls.pdf) for more detail.

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


class _RidgeRegression:
    """A simple class for ridge regression with closed-form solution."""

    def __init__(self, alpha=1.0):
        """Initialize the ridge regression model with the given regularization strength $\alpha$."""
        self.alpha = alpha
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ridge regression model to the given data. Compute linear weights (with intercept)
        of shape `(n_features + 1, n_targets)`.

        $w = (X^T X + \alpha I)^{-1} X^T y$

        :param X: the design matrix of shape `(n_samples, n_features)`
        :param y: the target values of shape `(n_samples, n_targets)`
        """
        n_samples, n_features = X.shape

        # Add bias term (column of ones) to the design matrix for intercept
        X_bias = np.hstack([np.ones((n_samples, 1)), X])

        # Regularization matrix (identity matrix with top-left value zero for intercept term)
        identity = np.eye(n_features + 1)
        identity[0, 0] = 0

        # Closed-form solution (normal equation) for ridge regression
        A = X_bias.T @ X_bias + self.alpha * identity
        B = X_bias.T @ y
        self.weights = np.linalg.solve(A, B)

    def predict(self, X: np.ndarray):
        """Compute the predicted target values for the given input data.

        :param X: the input data of shape `(n_samples, n_features)`
        :returns: the predicted target values of shape `(n_samples, n_targets)`
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        n_samples, n_features = X.shape

        # Add bias term (column of ones) to the design matrix for intercept
        X_bias = np.hstack([np.ones((n_samples, 1)), X])

        return X_bias @ self.weights


def _inspect_function(func):
    """Try to inspect the inputs and outputs of a callable function.

    !!! Example
        ```python
        def my_func(a, b, c, **kwargs):
            # Do something
            return y1, y2

        _inspect_function(my_func)
        # Returns (['a', 'b', 'c'], ['y1', 'y2'])
        ```

    :param func: The callable function to inspect.
    :returns: A tuple of the positional arguments and return values of the function.
    """
    try:
        sig = inspect.signature(func)
        pos_args = [param.name for param in sig.parameters.values() if param.default == param.empty
                    and param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY)]

        try:
            source = inspect.getsource(func).strip()
        except OSError:
            from dill.source import getsource  # inspect in IDLE using dill
            source = getsource(func).strip()

        tree = ast.parse(source)

        # Find the return values
        class ReturnVisitor(ast.NodeVisitor):
            def __init__(self):
                self.return_values = []

            def visit_Return(self, node):
                if isinstance(node.value, ast.Tuple):
                    self.return_values = [elt.id for elt in node.value.elts]
                elif isinstance(node.value, ast.Name):
                    self.return_values = [node.value.id]
                else:
                    self.return_values = []

        return_visitor = ReturnVisitor()
        return_visitor.visit(tree)

        return pos_args, return_visitor.return_values
    except Exception:
        return [], []


def _inspect_assignment(class_name: str, stack_idx: int = 2) -> str | None:
    """Return the left-hand side of an assignment like "x = class_name(...)".

    !!! Example
        ```python
        class MyClass:
            def __init__(self):
                self.name = _inspect_assignment('MyClass')
        obj = MyClass()
        print(obj.name)
        # Output: 'obj'
        ```

    This function will do it's best to only return single assignments (i.e. `x = MyClass()`) and not more
    complex expressions like list comprehension or tuple unpacking. If the assignment is not found or an error occurs
    during inspection, it will return `None`.

    :param class_name: the name of the class that is being constructed
    :param stack_idx: the index of the stack frame to inspect (default is 2 since you likely call this from
                      inside the class constructor, so you need to go back one more frame from that to find
                      the original assignment caller).
    :returns: the variable name assigned to the class constructor (or `None`)
    """
    variable_name = None
    try:
        stack = inspect.stack()
        frame_info = stack[stack_idx]
        code_line = frame_info.code_context[frame_info.index].strip()
        parsed_code = ast.parse(code_line)
        if isinstance(parsed_code.body[0], ast.Assign):
            assignment = parsed_code.body[0]
            if len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name):
                target_name = assignment.targets[0].id
                if isinstance(assignment.value, ast.Call) and isinstance(assignment.value.func, ast.Name):
                    if assignment.value.func.id == class_name:
                        variable_name = target_name
    except Exception:
        variable_name = None
    finally:
        # del current_frame, caller_frame
        return variable_name


def _get_yaml_path(yaml_obj: yaml.Loader | yaml.Dumper):
    """Get the path to the YAML file being loaded or dumped."""
    try:
        save_path = Path(yaml_obj.stream.name).parent
        save_file = Path(yaml_obj.stream.name).with_suffix('')
    except Exception:
        save_path = Path('.')
        save_file = 'yaml'
    return save_path, save_file


def search_for_file(filename: str | Path, search_paths=None):
    """Search for the given filename in the current working directory and any additional search paths provided.

    :param filename: the filename to search for
    :param search_paths: paths to try and find the file in
    :returns: the full path to the file if found, otherwise the original `filename`
    """
    if not isinstance(filename, str | Path):
        return filename

    search_paths = search_paths or []
    search_paths.append('.')

    save_file = Path(filename)
    need_to_search = True
    try:
        need_to_search = ((len(save_file.parts) == 1 and len(save_file.suffix) > 0) or
                          (len(save_file.parts) > 1 and not save_file.exists()))
    except Exception:
        need_to_search = False

    # Search for the save file if it was a valid path and does not exist
    if need_to_search:
        found_file = False
        name = save_file.name
        for path in search_paths:
            if (pth := Path(path) / name).exists():
                filename = pth.resolve().as_posix()
                found_file = True
                break
        if not found_file:
            pass  # Let the caller handle the error (just return the original filename back to caller)
            # raise FileNotFoundError(f"Could not find save file '{filename}' in paths: {search_paths}.")

    return filename


def format_inputs(inputs: Dataset, var_shape: dict = None) -> tuple[Dataset, tuple[int, ...]]:
    """Broadcast and reshape all inputs to the same shape. Loop shape is inferred from broadcasting the leading dims
    of all input arrays. Input arrays are broadcast to this shape and then flattened.

    !!! Example
        ```python
        inputs = {'x': np.random.rand(10, 1, 5), 'y': np.random.rand(1, 1), 'z': np.random.rand(1, 20, 3)}
        fmt_inputs, loop_shape = format_inputs(inputs)
        # Output: {'x': np.ndarray(200, 5), 'y': np.ndarray(200,), 'z': np.ndarray(200, 3)}, (10, 20)
        ```

    :param inputs: `dict` of input arrays
    :param var_shape: `dict` of expected input variable shapes (i.e. for field quantities); assumes all inputs are 1d
                      if None or not specified (i.e. scalar)
    :returns: the reshaped inputs and the common loop shape
    """
    var_shape = var_shape or {}

    def _common_shape(shape1, shape2):
        """Find the common leading dimensions between two shapes (with np broadcasting rules)."""
        min_len = min(len(shape1), len(shape2))
        common_shape = []
        for i in range(min_len):
            if shape1[i] == shape2[i]:
                common_shape.append(shape1[i])
            elif shape1[i] == 1:
                common_shape.append(shape2[i])
            elif shape2[i] == 1:
                common_shape.append(shape1[i])
            else:
                break
        return tuple(common_shape)

    def _shorten_shape(name, array):
        """Remove extra variable dimensions from the end of the array shape (i.e. field quantity dimensions)."""
        shape = var_shape.get(name, None)
        if shape is not None and len(shape) > 0:
            if len(shape) > len(array.shape):
                raise ValueError(f"Variable '{name}' shape {shape} is longer than input array shape {array.shape}. "
                                 f"The input array for '{name}' should have at least {len(shape)} dimensions.")
            return array.shape[:-len(shape)]
        else:
            return array.shape

    # Get the common "loop" dimensions from all input arrays
    inputs = {name: np.atleast_1d(value) for name, value in inputs.items()}
    name, array = next(iter(inputs.items()))
    loop_shape = _shorten_shape(name, array)
    for name, array in inputs.items():
        array_shape = _shorten_shape(name, array)
        loop_shape = _common_shape(loop_shape, array_shape)
        if not loop_shape:
            break
    N = np.prod(loop_shape)
    common_dim_cnt = len(loop_shape)

    # Flatten and broadcast all inputs to the common shape
    ret_inputs = {}
    for var_id, array in inputs.items():
        if common_dim_cnt > 0:
            broadcast_shape = np.broadcast_shapes(loop_shape, array.shape[:common_dim_cnt])
            broadcast_shape += array.shape[common_dim_cnt:]
            ret_inputs[var_id] = np.broadcast_to(array, broadcast_shape).reshape((N, *array.shape[common_dim_cnt:]))
        else:
            ret_inputs[var_id] = array

    return ret_inputs, loop_shape


def format_outputs(outputs: Dataset, loop_shape: tuple[int, ...]) -> Dataset:
    """Reshape all outputs to the common loop shape. Loop shape is as obtained from a call to `format_inputs`.
    Assumes that all outputs are the same along the first dimension. This first dimension gets reshaped back into
    the `loop_shape`.

    !!! Example
        ```python
        outputs = {'x': np.random.rand(10, 1, 5), 'y': np.random.rand(10, 1), 'z': np.random.rand(10, 20, 3)}
        loop_shape = (2, 5)
        fmt_outputs = format_outputs(outputs, loop_shape)
        # Output: {'x': np.ndarray(2, 5, 1, 5), 'y': np.ndarray(2, 5, 1), 'z': np.ndarray(2, 5, 20, 3)}
        ```

    :param outputs: `dict` of output arrays
    :param loop_shape: the common leading dimensions to reshape the output arrays to
    :returns: the reshaped outputs
    """
    output_dict = {}
    for key, val in outputs.items():
        val = np.atleast_1d(val)
        output_dict[key] = val.reshape(loop_shape + val.shape[1:])  # Assumes (N, ...) output shape to start with
    return output_dict


def _tokenize(args_str: str) -> list[str]:
    """
    Helper function to extract tokens from a string of arguments while respecting nested structures.

    This function processes a string of arguments and splits it into individual tokens, ensuring that nested
    structures such as parentheses, brackets, and quotes are correctly handled.

    :param args_str: The string of arguments to tokenize
    :return: A list of tokens extracted from the input string

    !!! Example
        ```python
        args_str = "func(1, 2), {'key': 'value'}, [1, 2, 3]"
        _tokenize(args_str)
        # Output: ['func(1, 2)', "{'key': 'value'}", '[1, 2, 3]']
        ```
    """
    if args_str is None or len(args_str) == 0:
        return []
    tokens = []
    current_token = []
    brace_depth = 0
    in_string = False

    i = 0
    while i < len(args_str):
        char = args_str[i]
        if char in ('"', "'") and (i == 0 or args_str[i - 1] != '\\'):  # Toggle string state
            in_string = not in_string
            current_token.append(char)
        elif in_string:
            current_token.append(char)
        elif char in '([{':
            brace_depth += 1
            current_token.append(char)
        elif char in ')]}':
            brace_depth -= 1
            current_token.append(char)
        elif char == ',' and brace_depth == 0:
            if current_token:
                tokens.append(''.join(current_token).strip())
                current_token = []
        else:
            current_token.append(char)
        i += 1

    # Add last token
    if current_token:
        tokens.append(''.join(current_token).strip())

    return tokens


def parse_function_string(call_string: str) -> tuple[str, list, dict]:
    """Convert a function signature like `func(a, b, key=value)` to name, args, kwargs.

    :param call_string: a function-like string to parse
    :returns: the function name, positional arguments, and keyword arguments
    """
    # Regex pattern to match function name and arguments
    pattern = r"(\w+)(?:\((.*)\))?"
    match = re.match(pattern, call_string.strip())

    if not match:
        raise ValueError(f"Function string '{call_string}' is not valid.")

    # Extracting name and arguments section
    name = match.group(1)
    args_str = match.group(2)

    # Regex to split arguments respecting parentheses and quotes
    # arg_pattern = re.compile(r'''((?:[^,'"()\[\]{}*]+|'[^']*'|"(?:\\.|[^"\\])*"|\([^)]*\)|\[[^\]]*\]|\{[^{}]*\}|\*)+|,)''')  # noqa: E501
    # pieces = [piece.strip() for piece in arg_pattern.findall(args_str) if piece.strip() != ',']
    pieces = _tokenize(args_str)

    args = []
    kwargs = {}
    keyword_only = False

    for piece in pieces:
        if piece == '/':
            continue
        elif piece == '*':
            keyword_only = True
        elif '=' in piece and (piece.index('=') < piece.find('{') or piece.find('{') == -1):
            key, val = piece.split('=', 1)
            kwargs[key.strip()] = ast.literal_eval(val.strip())
            keyword_only = True
        else:
            if keyword_only:
                raise ValueError("Positional arguments cannot follow keyword arguments.")
            args.append(ast.literal_eval(piece))

    return name, args, kwargs


def relative_error(pred, targ, axis=None, skip_nan=False):
    """Compute the relative L2 error between two vectors along the given axis.

    :param pred: the predicted values
    :param targ: the target values
    :param axis: the axis along which to compute the error
    :param skip_nan: whether to skip NaN values in the error calculation
    :returns: the relative L2 error
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        sum_func = np.nansum if skip_nan else np.sum
        err = np.sqrt(sum_func((pred - targ)**2, axis=axis) / sum_func(targ**2, axis=axis))
    return np.nan_to_num(err, nan=np.nan, posinf=np.nan, neginf=np.nan)


def get_logger(name: str, stdout: bool = True, log_file: str | Path = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a file/stdout logger with the given name.

    :param name: the name of the logger to return
    :param stdout: whether to add a stdout stream handler to the logger
    :param log_file: add file logging to this file (optional)
    :param level: the logging level to set
    :returns: the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(level)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger
