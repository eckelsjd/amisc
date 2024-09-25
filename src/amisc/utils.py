"""Provides some basic utilities for the package.

Includes:

- `as_tuple` — convert a tuple-like object to a tuple of `ints`
- `parse_function_string` — convert function-like strings to arguments and keyword-arguments
- `relative_error` — compute the relative L2 error between two vectors
- `get_logger` — logging utility with nice formatting
"""
import ast
import logging
import re
import sys
from pathlib import Path

import numpy as np

LOG_FORMATTER = logging.Formatter(u"%(asctime)s — [%(levelname)s] — %(name)-25s — %(message)s")


def as_tuple(value: str | tuple) -> tuple[int, ...]:
    """Convert a tuple-like object to a tuple of `ints`."""
    if isinstance(value, str):
        return tuple([int(i) for i in ast.literal_eval(value)])
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
