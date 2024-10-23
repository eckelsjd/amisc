# Things that need documentation

- Serialization mixins and other utilities
- Variable data structures and transform/distribution/compression classes
- Specifying component models and variables in yaml (and saving/loading)
- Interpolators, States, and TrainingData ABCs and default implementation for SG + Lagrange
- Specifying custom serializers for component data (interpolator, states, training data, etc.) -- can specify these in construction, use defaults, or allow auto-detection from types of other Component(**kwargs), or dict specification in yaml config
- Model wrapper call_model() -- only top-level callables, also inspection of args and unpacked calling
- Might want custom args/kwargs/interp/training classes so you can have custom serialization/methods. Each will implement
an interface that does common things (and may inherit default serialization). Each interface will provide a from_dict
method to additionally allow constructing the custom class from a dict (which would be used for loading from a config file).
The custom classes will then be stored in component.serializers, from which every subsequent save to file will serialize
with the custom methods.
- tutorial notebooks
- extra model args only specified/saved as model_kwargs, can use Component(model_kwargs={}) or Component(**kwargs)
- All model args are inputs. Must pass outputs=[]
- dict args to predict -- specify pass-through values for components
- variable field quantity should have a list of domains for each latent coeff
- TrainingData should account for model errors, np.nan imputation, beta initialization at (0,0,...), var normalization, and field qtys
- usage of shape, list domain, and compression for field quantities
- transforms do not apply to latent coeff
- refining a field qty input should give a full tensor-product upgrade of its latent coeff (or round-robin)
- model can return y_var_coords along with full fields y_var_XX aligning with var.compression.fields
- model can request {var.name}_coords in kwargs and get them from call_model() during activate_index
- can also request specific coords in model_kwargs, and those will be passed through call_model() after interpolation
- LATENT_STR_ID -- training_Data/interpolator can import this to handle latent coeff variables in a special way _if they want to_
- vectorized alpha will be np.array (*shape, len(alpha))


Serialization
- serialize/deserialize are very general to/from Python builtins for amisc classes
- yaml_representer/constructor are encoder/decoders specific to the pyyaml library
- yaml_load/yaml_dump are convenience functions for using the yaml encoder/decoders defined in amisc
- save_to_file / load_from_file are convenience functions for the high-level SystemSurrogate that takes advantage of assumed amisc directory structure
- You could implement any encoder/decoders or convenience functions external to amisc (i.e. json) and make use of the low-level serialize/deserialize functions. The yaml functions are just the recommended and amisc-builtin methods for serialization
