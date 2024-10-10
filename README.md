![Logo](https://raw.githubusercontent.com/eckelsjd/amisc/main/docs/assets/amisc_logo_text.svg)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
[![PyPI](https://img.shields.io/pypi/v/amisc?logo=python&logoColor=%23cccccc)](https://pypi.org/project/amisc)
![build](https://img.shields.io/github/actions/workflow/status/eckelsjd/amisc/deploy.yml?logo=github)
![docs](https://img.shields.io/github/actions/workflow/status/eckelsjd/amisc/docs.yml?logo=materialformkdocs&logoColor=%2523cccccc&label=docs)
![tests](https://img.shields.io/github/actions/workflow/status/eckelsjd/amisc/tests.yml?logo=github&logoColor=%2523cccccc&label=tests)
![Code Coverage](https://img.shields.io/badge/coverage-47%25-orange?logo=codecov)
[![Algorithm description](https://img.shields.io/badge/DOI-10.1002/nme.6958-blue)](https://doi.org/10.1002/nme.6958)

Efficient framework for building surrogates of multidisciplinary systems using the adaptive multi-index stochastic collocation ([AMISC](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6958))  technique.

## NOTICE: this branch is still undergoing major changes
Expect most things to not be working yet (documentation is also not up-to-date everywhere)

## ⚙️ Installation
```shell
pip install amisc
```
If you are using [pdm](https://github.com/pdm-project/pdm) in your own project, then you can use:
```shell
pdm add amisc

# Or in editable mode from a local clone...
pdm add -e ./amisc --dev
```

## 📍 Quickstart
```python
import numpy as np

from amisc import System

def fun1(x):
    y1 = x * np.sin(np.pi * x)
    return y1

def fun2(y1):
    y2 = 1 / (1 + 25 * y1 ** 2)
    return y2

system = System(fun1, fun2)
system.fit()

x_test = system.sample_inputs(10)
y_test = system.predict(x_test)
```

## 🏗️ Contributing
See the [contribution](https://github.com/eckelsjd/amisc/blob/main/CONTRIBUTING.md) guidelines.

## 📖 Reference
AMISC paper [[1](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6958)].

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
