Lagrange polynomials are used in `amisc` for constructing a smooth interpolation surface over a set of training data. The Lagrange polynomial is a fundamental tool in numerical interpolation, allowing for polynomial approximation of functions based on known data points.

## Univariate Lagrange polynomials

For a set of \( n+1 \) distinct data points \((x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)\), the Lagrange polynomial \( L(x) \) is defined as:

\[
L(x) = \sum_{i=0}^{n} y_i \ell_i(x)
\]

where each basis polynomial \(\ell_i(x)\) is given by:

\[
\ell_i(x) = \prod_{\substack{0 \le j \le n \\ j \neq i}} \frac{x - x_j}{x_i - x_j}
\]

This formula ensures that \( \ell_i(x_j) = \delta_{ij} \), where \(\delta_{ij}\) is the Kronecker delta, making \(\ell_i(x)\) equal to 1 at \( x_i \) and 0 at all other \( x_j \).

## Multivariate Lagrange polynomials

In higher dimensions, the interpolation of a function over a grid of points can be achieved using a tensor-product extension of the univariate Lagrange polynomials. For a multi-dimensional input \( \mathbf{x} = (x_1, x_2, \ldots, x_d) \), the tensor-product Lagrange polynomial \( L(\mathbf{x}) \) is constructed as follows:

\[
L(\mathbf{x}) = \sum_{i_1=0}^{n_1} \sum_{i_2=0}^{n_2} \cdots \sum_{i_d=0}^{n_d} f_{i_1,i_2,\ldots,i_d} \prod_{j=1}^{d} \ell_{i_j}(x_j)
\]

where each \(\ell_{i_j}(x_j)\) is the univariate Lagrange basis polynomial for the \( j \)-th dimension, and \( f_{i_1,i_2,\ldots,i_d} \) are the values of the function at the grid points in the multi-dimensional space.

## Usage in `amisc`

In the context of `amisc`, Lagrange polynomials are used to construct single-fidelity surrogates \(f_{k, (\alpha, \beta)}(x)\) by evaluating the \(\alpha\)-fidelity model at a Cartesian grid defined on the parametric domain. This approach efficiently interpolates data in multiple dimensions, taking advantage of the structure provided by tensor-product grids.