![Logo](https://raw.githubusercontent.com/eckelsjd/amisc/main/docs/assets/amisc_logo_text.svg)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![PyPI](https://img.shields.io/pypi/v/amisc?logo=python&logoColor=%23cccccc)](https://pypi.org/project/amisc)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
![Commits](https://img.shields.io/github/commit-activity/m/eckelsjd/amisc?logo=github)
![build](https://img.shields.io/github/actions/workflow/status/eckelsjd/amisc/deploy.yml?logo=github
)
![docs](https://img.shields.io/github/actions/workflow/status/eckelsjd/amisc/docs.yml?logo=materialformkdocs&logoColor=%2523cccccc&label=docs)
![tests](https://img.shields.io/github/actions/workflow/status/eckelsjd/amisc/tests.yml?logo=github&logoColor=%2523cccccc&label=tests)
[![Coverage Status](https://coveralls.io/repos/github/eckelsjd/amisc/badge.svg?branch=main)](https://coveralls.io/github/eckelsjd/amisc?branch=main)
[![Algorithm description](https://img.shields.io/badge/DOI-10.1002/nme.6958-blue)](https://doi.org/10.1002/nme.6958)

Efficient framework for building surrogates of multidisciplinary systems. 
Uses the adaptive multi-index stochastic collocation ([AMISC](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6958)) 
technique.

## Installation
We highly recommend using [pdm](https://github.com/pdm-project/pdm):
```shell
pip install --user pdm
pdm add amisc
```
However, you can also install normally:
```shell
pip install amisc
```
To install from an editable local directory (e.g. for development), first fork the repo and then:
```shell
git clone https://github.com/<your-username>/amisc.git
pdm add -e ./amisc --dev  # or..
pip install -e ./amisc    # similarly
```
This way you can make changes to `amisc` locally while working on some other project for example.
You can also quickly set up a dev environment with:
```shell
git clone https://github.com/<your-username>/amisc.git
cd amisc
pdm sync  # reads pdm.lock and sets up an identical venv
```

## Quickstart
```python title="amisc.examples.tutorial.py"
--8<-- "amisc/examples/tutorial.py:simple"
```

## Contributing
See the [contribution](CONTRIBUTING.md) guidelines.

## Citations
AMISC paper [[1](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6958)].

