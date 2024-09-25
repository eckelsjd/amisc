- will need a way to set the interpolator state serializer ahead of time (extra arg to component? -- or misc_states get filled automatically during training and the state is inferred from the type check)
- will need a convenience function for saving/loading from a single yaml file that references pickle save files

Places where scalar input assumption is made:
- component.call_model(), xdim
- Variable - dist, transform, nominal, domain, serialize
-
