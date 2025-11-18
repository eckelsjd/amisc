## v0.8.1 (2025-11-18)

### Fix

- fixes #58 loop dim of (1,) gets squeezed for field qtys
- fixes #53 compression validation

## v0.8.0 (2025-05-15)

### Feat

- adds GPR for interpolation
- adds scaler opt for linear regression
- adds linear regression Interpolator

## v0.7.3 (2025-02-10)

### Fix

- fixes #45 typo in call_model
- handle None/nan samples during compression

## v0.7.2 (2025-02-03)

### Fix

- fixes indexing error in system predict when handling exceptions

## v0.7.1 (2025-01-23)

### Fix

- all component outputs returned by system predict

## v0.7.0 (2025-01-16)

### Feat

- adds training data caching and performance improvements

### Fix

- makes test the default index set for test set performance
- bug with error indicator model cost

## v0.6.0 (2025-01-09)

### Feat

- clean sample inputs and plot slice api
- overhead tracking and sorted fidelity evaluation
- adds flag for custom weighting functions during training
- adds reconstruction tolerance for svd

### Fix

- bug with component serializer dict validation
- more consistent tracking of model costs and allocation
- rank starts at 1 for svd reconstruction tolerance
- serialize system model_extra if builtin
- allow skipping nan in relative error
- fixes bug in load_from_file and allows nd object arrays for compression
- cushion turbojet test tolerance
- integrate field coords and object arrays for changing field shapes

### Refactor

- update PEP735 pyproject dependency groups
- rename sweep plots to slice

## v0.5.2 (2024-11-05)

### Fix

- fixes interactive function inspection and cleans dependencies

## v0.5.1 (2024-11-04)

### Fix

- init versioning and gh release

## v0.5.0 (2024-11-04)

### BREAKING CHANGE

- mostly all APIs affected, resolves #23, #20, #19, #17, #16, #12, #11, #10, and JANUS-Institute/HallThrusterPEM#6
- Remove BaseRV interface and rename rv.py to variable.py

### Feat

- completes overhaul to v0.5.0
- adds component pydantic validation, serialization, and model wrapper
- add pydantic validation to Variable, add Transform and Distribution abstractions
- replace BaseRV with Variable and normalization and compression

## v0.4.0 (2024-08-29)

### Feat

- migrate to copier-numpy template
