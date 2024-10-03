- will need a way to set the interpolator state serializer ahead of time (extra arg to component? -- or misc_states get filled automatically during training and the state is inferred from the type check)

- swap component.serializers interactively (i.e. you have new model args that need custom serialization or new training data storage) -- will need a way to transfer the data over to the new type

Places where scalar input assumption is made:
- Variable - dist, transform, nominal, domain, serialize


- component.predict(use_model, output_path, index_Set)