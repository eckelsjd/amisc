The `amisc` library handles multidisciplinary systems with feedback coupling using a fixed-point iteration (FPI) routine. This approach is crucial for solving systems where the outputs of some components are inputs to others, and vice versa, forming cycles of dependencies.

## Feedback coupling

In multidisciplinary systems, feedback coupling occurs when the outputs of one component influence the inputs of another and vice versa, forming cycles. These cycles can be represented as strongly-connected components (SCCs) in a graph-theoretic representation of the system. Each SCC represents a group of components that are interdependent.

!!! Example
    In the following system of 3 nonlinear equations: $f_1, f_2, f_3$, the $(f_2, f_3)$ group forms an SCC with coupling variables $(y_2, y_3)$. Viewing the condensation of the system graph using the `networkx` library highlights this feedback cycle.
    ```python
    from amisc import System
    import networkx as nx
    
    def f1(x):
        y1 = x ** 2
        return y1
    
    def f2(y1, y3):
        y2 = y1 + np.sin(y3)
        return y2

    def f3(y2):
        y3 = np.cos(y2)
        return y3

    system = System(f1, f2, f3)
    dag = nx.condensation(system.graph())  # Group SCCs to form a directed-acyclic-graph
    
    for idx, node in dag.nodes.items():
        print(node['members'])

    # gives 'f1' and ['f2', 'f3']
    ```

## Fixed-Point Iteration (FPI)

An FPI routine is employed to solve for the coupling variables within each SCC. The process involves iteratively updating the coupling variables until convergence is achieved. The steps are as follows:

1. __Initialization__ - Begin with an initial guess for the coupling variables \(\xi^{(0)}\) within the SCC.

2. __Iterative update__ - For each iteration \( i \), update the coupling variables \(\xi^{(i)}\) using the function \( F(\xi^{(i-1)}) \), which evaluates the component models in the SCC using the current estimates of the coupling variables:
        \begin{equation}
        \xi^{(i)} = F(\xi^{(i-1)}).
        \end{equation}
3. __Convergence check__ - Continue iterating until the change in the coupling variables is below a specified tolerance, i.e., \( \lVert\xi^{(i)} - \xi^{(i-1)}\rVert < \eta \), where \(\eta\) is the convergence tolerance.

## Usage in `amisc`

When using `amisc` surrogates for prediction, the system's graph structure is divided into SCCs, and each SCC is solved independently using the FPI routine with the surrogate models. The feedback dependencies are automatically managed under-the-hood during prediction, and `amisc` allows the user to specify some keywords to fine-tune the FPI routine, including the convergence tolerance \(\eta\) and the maximum number of iterations.

!!! Example
    Using the three-component system of $(f_1, f_2, f_3)$ defined above, predictions are obtained with the surrogate `predict()` method, with FPI options passed as keywords.
    ```python
    system = System(f1, f2, f3)  # defined in previous example
    
    system.predict({'x': 0.5}, max_fpi_iter=100, fpi_tol=1e-10, andersom_mem=5)
    ```

Note that the `System` implements the [Anderson acceleration](https://doi.org/10.1137/10078356X) algorithm for enhancing the convergence of FPI.