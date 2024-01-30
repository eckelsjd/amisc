"""`utils.py`

Provides some basic utilities for the package.

Includes
--------
- `load_variables`: convenience function for loading RVs from a .json config file
- `get_logger`: logging utility with nice formatting
"""
import json
from pathlib import Path
import logging
import sys

from amisc.rv import BaseRV, UniformRV, NormalRV, ScalarRV


LOG_FORMATTER = logging.Formatter("%(asctime)s \u2014 [%(levelname)s] \u2014 %(name)-25s \u2014 %(message)s")


def load_variables(variables: list[str], file: Path | str) -> list[BaseRV]:
    """Load a list of BaseRV objects from a variables json `file`.

    :param variables: a list of str ids for variables to find in `file`
    :param file: json file to search for variable definitions
    :returns rvs: a list of corresponding `BaseRV` objects
    """
    with open(Path(file), 'r') as fd:
        data = json.load(fd)

    rvs = []
    keys = ['id', 'tex', 'description', 'units', 'param_type', 'nominal', 'domain']
    for str_id in variables:
        if str_id in data:
            var_info = data.get(str_id)
            kwargs = {key: var_info.get(key) for key in keys if var_info.get(key)}
            match var_info.get('rv_type', 'none'):
                case 'uniform_bds':
                    bds = var_info.get('rv_params')
                    rvs.append(UniformRV(bds[0], bds[1], **kwargs))
                case 'uniform_pct':
                    rvs.append(UniformRV(var_info.get('rv_params'), 'pct', **kwargs))
                case 'uniform_tol':
                    rvs.append(UniformRV(var_info.get('rv_params'), 'tol', **kwargs))
                case 'normal':
                    mu, std = var_info.get('rv_params')
                    rvs.append(NormalRV(mu, std, **kwargs))
                case 'none':
                    # Make a plain stand-in scalar RV object (no uncertainty)
                    rvs.append(ScalarRV(**kwargs))
                case other:
                    raise NotImplementedError(f'RV type "{var_info.get("rv_type")}" is not known.')
        else:
            raise ValueError(f'You have requested the variable {str_id}, but it was not found in {file}. '
                             f'Please add a definition of {str_id} to {file} or construct it on your own.')

    return rvs


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
