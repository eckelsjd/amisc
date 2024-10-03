# Things that need documentation

- Serialization mixins and other utilities
- Variable data structures and transform/distribution/compression classes
- Specifying component models and variables in yaml (and saving/loading)
- Interpolators, States, and TrainingData ABCs and default implementation for SG + Lagrange
- Specifying custom serializers for component data (interpolator, states, training data, etc.) -- can specify these in construction, use defaults, or allow auto-detection from types of other Component(**kwargs), or dict specification in yaml config
- Model wrapper call_model() -- only top-level callables
- Might want custom args/kwargs/interp/training classes so you can have custom serialization/methods. Each will implement
an interface that does common things (and may inherit default serialization). Each interface will provide a from_dict
method to additionally allow constructing the custom class from a dict (which would be used for loading from a config file).
The custom classes will then be stored in component.serializers, from which every subsequent save to file will serialize
with the custom methods.


Serialization
- serialize/deserialize are very general to/from Python builtins for amisc classes
- yaml_representer/constructor are encoder/decoders specific to the pyyaml library
- yaml_load/yaml_dump are convenience functions for using the yaml encoder/decoders defined in amisc
- save_to_file / load_from_file are convenience functions for the high-level SystemSurrogate that takes advantage of assumed amisc directory structure
- You could implement any encoder/decoders or convenience functions external to amisc (i.e. json) and make use of the low-level serialize/deserialize functions. The yaml functions are just the recommended and amisc-builtin methods for serialization
