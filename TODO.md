- test field qtys
- test convergence
- system plots
- system documentation
- examples documentation and README
- important things from TODOC
- make sure doc site works
- clean up dependencies and lock
- pre commit checks
- version/bump and merge

- system simulate training
- system retroactively add outputs (load training data from saved files)
- make sure thread pool works/is efficient on refine

- profile memory/time of predict and fit for various executors and input sizes
- optimize predict() mainly (fit will be overwhelmed by model costs)