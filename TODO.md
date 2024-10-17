- swap component.serializers interactively (i.e. you have new model args that need custom serialization or new training data storage) -- will need a way to transfer the data over to the new type

Places where scalar input assumption is made:
- Variable - dist, transform, nominal, domain, serialize

- simulate training
- retroactively add outputs (load training data from saved files)

- make sure thread pool works/is efficient on refine