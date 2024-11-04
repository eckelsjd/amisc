The `amisc` Python library provides tools for constructing multi-fidelity surrogates for multidisciplinary systems, based on methodologies outlined in the paper titled

"Adaptive Experimental Design for Multi-Fidelity Surrogate Modeling of Multi-Disciplinary Systems" 

by [Jakeman et al](https://onlinelibrary.wiley.com/doi/10.1002/nme.6958). This library leverages the Multi-Index Stochastic Collocation (MISC) approach to efficiently approximate complex system models by combining evaluations of varying fidelity. 

A "multi-index" is a tuple of whole numbers that defines a set of fidelity levels for a model, ranging from the lowest fidelity $(0, 0, \dots)$ to the highest fidelity $(N_1, N_2, \dots)$. "Stochastic collocation" refers to the process of choosing collocation points (i.e. knots, grid points, training data, etc.) for a set of random variable inputs for the purposes of function approximation or moment estimation.

Model fidelity is collectively defined by a pair of multi-indices:

- $\alpha$ - Specifies the deterministic fidelity of a model. It controls the mesh size, time step, and other hyper-parameters that influence the physical model's resolution and accuracy.
- $\beta$ - Specifies the parametric fidelity of a surrogate. It dictates the number of samples used to construct the surrogate and the complexity of the surrogate itself, impacting its computational cost and accuracy.

## Key Ideas

- __Multi-Index Stochastic Collocation (MISC)__ - The library implements MISC to create multi-fidelity surrogates by combining multiple model fidelities. The surrogate model for each component $k$ in a multidisciplinary system is expressed as a linear combination of multiple surrogates:

  \begin{equation}
  f_{k}(x) \approx \sum_{(\alpha, \beta)\in\mathcal{I_k}} c_{k, (\alpha, \beta)} f_{k, (\alpha, \beta)}(x).
  \end{equation}

  Here, $f_{k, (\alpha, \beta)}(x)$ represents the single-fidelity surrogate for a given fidelity level $(\alpha, \beta)$, and $\mathcal{I}_k$ is a set of concatenated multi-indices specifying different fidelities.

- __Downward-closed index set__ - The set $\mathcal{I}_k$ must be downward-closed to ensure that if $(\gamma, \delta) \leq (\alpha, \beta)$ and $(\alpha, \beta) \in \mathcal{I}_k$, then $(\gamma, \delta) \in \mathcal{I}_k$.

- __Combination coefficients__ - The coefficients for the MISC approximation are calculated using:

  \begin{equation}
  c_{k, (\alpha, \beta)} = \sum_{(\alpha + i, \beta + j)\in \mathcal{I}_k} (-1)^{\lVert i, j\rVert _1},
  \end{equation}
    
  for all $(i, j)\in(0, 1)^{\mathrm{len}(\alpha, \beta)}$. That is, we sum over every $(\alpha, \beta)$ in the index set for which the _lower neighbor_ is also in the index set (i.e. subtract one from every index number and check if it is in $\mathcal{I}_k$).

  This formula ensures efficient computation of the surrogate model by balancing deterministic and parametric errors.

- __Adaptive training__ - The adaptive training procedure involves refining the surrogate model by activating indices and searching for new refinement directions:

    - _Activated Indices_ - These are the indices in the set $\mathcal{I}_k$ that have been selected for inclusion in the surrogate model. The refinement process begins by identifying the index with the largest error indicator from the set of candidate indices.
    - _Candidate Indices_ - These indices are the nearest _forward neighbors_ of the active indices and represent potential directions for refinement. The algorithm searches over these indices to identify new refinement directions that can improve the surrogate model's accuracy. Each candidate index is evaluated based on its potential to reduce the error in the system-level quantities of interest.
    - _Refinement_ - Once a candidate index is selected, it is added to the activated set, and new training samples are generated. This iterative process continues, dynamically allocating computational resources to the most impactful components for minimizing prediction error.

## Further Reading

For a comprehensive understanding of the methodologies and theoretical underpinnings, please refer to the original paper by Jakeman et al., available at [DOI: 10.1002/nme.6958](https://onlinelibrary.wiley.com/doi/10.1002/nme.6958).
