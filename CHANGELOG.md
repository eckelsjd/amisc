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
