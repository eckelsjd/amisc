![](assets/amisc_logo_text.svg)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

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
```python
from amisc.surrogates import SystemSurrogate
from amisc.utils import UniformRV
import numpy as np

def fun1(x):
    return x ** 2

def fun2(y):
    return np.sin(y) * np.exp(y)

x, y, z = UniformRV(0, 1, 'x'), UniformRV(0, 1, 'y'), UniformRV(0, 1, 'z')
model1 = {'name': 'model1', 'model': fun1, 'exo_in': ['x'], 'coupling_out': ['y']}
model2 = {'name': 'model2', 'model': fun2, 'coupling_in': ['y'], 'coupling_out': ['z']}

system = SystemSurrogate([model1, model2], [x], [y, z])
system.fit()

xtest = system.sample_inputs(10)
ytest = system.predict(xtest)
```

## Contributing
See the [contribution](CONTRIBUTING.md) guidelines.

## Reference
AMISC paper [[1](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6958)].

