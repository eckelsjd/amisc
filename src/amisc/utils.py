"""Provides some basic utilities for the package.

Includes:

- `search_for_file` — search for a file in the current working directory and additional search paths
- `format_inputs` — broadcast and reshape all inputs to the same shape
- `format_outputs` — reshape all outputs to a common loop shape
- `as_tuple` — convert a tuple-like object to a tuple of `ints`
- `parse_function_string` — convert function-like strings to arguments and keyword-arguments
- `relative_error` — compute the relative L2 error between two vectors
- `get_logger` — logging utility with nice formatting
"""
import ast
import inspect
import logging
import re
import sys
from pathlib import Path

import numpy as np
import yaml

__all__ = ['as_tuple', 'parse_function_string', 'relative_error', 'get_logger', 'format_inputs', 'format_outputs',
           'search_for_file']

LOG_FORMATTER = logging.Formatter(u"%(asctime)s — [%(levelname)s] — %(name)-25s — %(message)s")


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
        # current_frame = inspect.currentframe()
        # caller_frame = current_frame.f_back
        # code_obj = caller_frame.f_code
        # line_number = caller_frame.f_lineno
        # function_source, first_line = inspect.getsourcelines(code_obj)
        # code_line = function_source[line_number - first_line].strip()
        parsed_code = ast.parse(code_line)
        if isinstance(parsed_code.body[0], ast.Assign):
            assignment = parsed_code.body[0]
            if len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name):
                target_name = assignment.targets[0].id
                if isinstance(assignment.value, ast.Call) and isinstance(assignment.value.func, ast.Name):
                    if assignment.value.func.id == class_name:
                        variable_name = target_name
    except:
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
        filename = save_file.name
        for path in search_paths:
            if (pth := Path(path) / filename).exists():
                filename = pth.resolve().as_posix()
                found_file = True
                break
        if not found_file:
            raise FileNotFoundError(f"Could not find save file '{filename}' in paths: {search_paths}.")

    return filename


def format_inputs(inputs: dict, var_shape: dict = None) -> tuple[dict, tuple[int, ...]]:
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
                      if None or not specified
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


def format_outputs(outputs: dict, loop_shape: tuple[int, ...]) -> dict:
    """Reshape all outputs to the common loop shape. Loop shape is as obtained from a call to `format_inputs`.
    Assumes that all outputs are the same along the first dimension. This first dimension gets reshaped back into
    the `loop_shape`. Singleton outputs are squeezed along the last dimension. A singleton loop shape is squeezed
    along the first dimension.

    !!! Example
        ```python
        outputs = {'x': np.random.rand(10, 1, 5), 'y': np.random.rand(10, 1), 'z': np.random.rand(10, 20, 3)}
        loop_shape = (2, 5)
        fmt_outputs = format_outputs(outputs, loop_shape)
        # Output: {'x': np.ndarray(2, 5, 1, 5), 'y': np.ndarray(2, 5), 'z': np.ndarray(200, 3)}, (2, 5, 20, 3)
        ```

    :param outputs: `dict` of output arrays
    :param loop_shape: the common leading dimensions to reshape the output arrays to
    :returns: the reshaped outputs
    """
    output_dict = {}
    for key, val in outputs.items():
        val = np.atleast_1d(val)
        output_shape = val.shape[1:]  # Assumes (N, ...) output shape to start with
        val = val.reshape(loop_shape + output_shape)
        if output_shape == (1,):
            val = np.atleast_1d(np.squeeze(val, axis=-1))  # Squeeze singleton outputs
        if loop_shape == (1,):
            val = np.atleast_1d(np.squeeze(val, axis=0))  # Squeeze singleton loop dimensions
        output_dict[key] = val
    return output_dict


def as_tuple(value: str | tuple | int) -> tuple[int, ...]:
    """Convert a tuple-like object to a tuple of `ints`."""
    if isinstance(value, str):
        return tuple([int(i) for i in ast.literal_eval(value)])
    else:
        if isinstance(value, int):
            return (value,)
        else:
            return tuple([int(i) for i in value])


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
    """Convert a function signature like `func(a, b, key=value)` to name, args, kwargs."""
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


def relative_error(pred, targ, axis=None):
    return np.sqrt(np.sum((pred - targ)**2, axis=axis) / np.sum(targ**2, axis=axis))


def get_logger(name: str, stdout=True, log_file: str | Path = None) -> logging.Logger:
    """Return a file/stdout logger with the given name.

    :param name: the name of the logger to return
    :param stdout: whether to add a stdout handler to the logger
    :param log_file: add file logging to this file (optional)
    :returns: the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger
