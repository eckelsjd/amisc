Configuration files can be written to define `amisc` objects using the text-based [YAML](https://pyyaml.org/) markup language. In general, YAML files contain mappings, sequences, and scalars, which correspond roughly to Python `dicts`, `lists`, and `strings`, respectively.

!!! Example "YAML configuration file"
    ```yaml
    # Mapping (dictionary)
    description: My grocery list
    store: Winco foods
    location: 1651 N 400 E
    date: 11-4-2024
    
    # Sequence (list)
    items:
    - Bananas
    - Bread
    - Milk
    - Eggs

    # Scalar (strings)
    notes: All values like this one are treated as strings
    block: |
        Strings can also be written in "block" form
        so that they may take up multiple lines.
    numbers: 1.0  # numbers are also strings but will get converted upon loading
    ```

More complicated objects can be defined using YAML "tags", which are demarcated by an exclamation followed by the name of the object: `!ObjectName`. We provide access to three object tags for defining `amisc` objects in YAML: `!Variable`, `!Component`, and `!System`. The [`YamlLoader`][amisc.YamlLoader] class contains rules for loading these tags into the `amisc` framework.

## Variables
Variable objects are constructed with the `!Variable` tag followed by a mapping of the variable properties:
```yaml
!Variable
name: x
description: My custom variable
domain: (0, 1)
nominal: 0.5
distribution: U(0, 1)
norm: minmax
units: m/s
```

A list of variables may be defined with the `!VariableList` tag:
```yaml
!VariableList
- name: x1
  domain: (0, 1)
- name: x2
  description: another variable
- name: x3
```

## Components
Component objects are constructed with the `!Component` tag followed by a mapping of the component properties. Lists of variable inputs and outputs may be defined by nesting the `!VariableList` tag:
```yaml
!Component
name: My component
model: !!python/name:my.importable.model_function
model_kwargs:
  extra_config: A config value
  options: More options here passed as **kwarg to the model_function
inputs: !VariableList
  - name: x1
  - name: x2
outputs: !VariableList
  - name: y1
  - name: y2
data_fidelity: (2, 2)
vectorized: true
```

A list of components can be constructed by listing several components in a sequence underneath the `!Component` tag:
```yaml
!Component
- name: First component
  model: !!python/name:amisc.examples.models.f1
- name: Second component
  model: !!python/name:amisc.examples.models.f2
```

!!! Note "Defining callable functions in YAML"
    In the examples above, we defined callable Python functions using the `!!python/name` tag followed by the import path of the function. The import path must be defined in a global scope so that a Python `import my.model_function` statement is valid. For example, you might define your function in a local importable package, or simply in the current working directory (which is always searched by the Python module finder). If you had a local `module.py` file that contained the `my_model` file, then you would specify this in YAML as `!!python/name:module.my_model`. 

## System
The `System` surrogate object is constructed with the `!System` tag followed by a mapping of the system properties. Lists of components may be defined by nesting the `!Component` tag:
```yaml
!System
name: My multidisciplinary system
components: !Component
  - name: My first component
    model: !!python/name:path.to.first_model
    inputs: !VariableList
      - name: x1
      - name: x2
    outputs: !VariableList
      - name: y1
      - name: y2
  - name: My second component
    model: !!python/name:path.to.second_model
```

!!! Note "Duplicate variables"
    If multiple components take the same input variable, you only need to define the variable once in the YAML file. Then, you may simply reference the variable's `name` for any other component that uses the variable. Upon loading from file, the `System` will use the same `Variable` object for all components that reference the same variable `name`.

## Loading from a configuration file
The [YamlLoader][amisc.YamlLoader] provides an interface for loading `amisc` objects from YAML config files:
```python
from amisc import YamlLoader

config_file = """
!VariableList
- name: x1
  domain: (0, 1)
- name: x2
- name: x3
"""

variables = YamlLoader.load(config_file)
```

The [`load_from_file`][amisc.system.System.load_from_file] and [`save_to_file`][amisc.system.System.save_to_file] convenience methods are also provided for loading and saving `System` objects to file (i.e. during surrogate training). These surrogate save files closely mirror the YAML format used for configuration -- they contain all the base properties of the surrogate as well as extra internal data for storing the state of the surrogate. These save files can be edited directly in a text editor to change or view properties of the surrogate.