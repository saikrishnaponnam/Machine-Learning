# Linear Algebra

## Notations
Scalar $x \in \mathbb{R}$  
Vector $ \mathbf{x} \in \mathbb{R}^n$

$$\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots\\
x_n
\end{bmatrix}
$$

Matrix $ \mathbf{X} \in \mathbb{R}^{n \times m} $

$$\mathbf{X} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1m} \\
x_{21} & x_{22} & \cdots & x_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nm}
\end{bmatrix}$$


We will use $\mathbf{X}$ to denote input matrix that contains training samples and features.

## Matrix Operations

**Hadamard product**: The elementwise product of two matrices, denoted by $\circ$.

<!-- Properties -->


## Linear Transformations
<!--  how matrices transform vector spaces.
Geometric interpretation: Rotation, scaling, shearing, projection. -->

## Vector Spaces and Subspaces
<!-- Span, basis, dimension, null space, column space, rank.
Application: Understanding solution spaces of linear systems; identifying independent features.
-->


## Eigenvalues and Eigenvectors
<!-- Diagonalization
Spectral decomposition -->
An eigenvector for a matrix $A$ is a nonzero vector $\mathbf{x}$ such that $A\mathbf{x} = c\mathbf{x}$, where c is some constant.
The constant $c$ is called the eigenvalue corresponding to the eigenvector $\mathbf{x}$.
The eigenvalue problem can be solved by finding the roots of the characteristic polynomial:

$$\text{det}(A - cI) = 0$$


## Matrix Factorizations
<!-- LU decomposition 
QR Decomposition
Singular Value Decomposition (SVD) -->

## Norms and Distance Metrics
<!--  Frobenius norms -->
**Vector Norms:**
The norm of a vector tells us how big it is.
A norm is a function $||.||$ that maps a vector to a scalar and satisfies the following three properties:

1. **Non-negativity**: $||\mathbf{x}|| \geq 0$ and $||\mathbf{x}|| = 0$ if and only if $\mathbf{x} = 0$.
2. **Homogeneity**: $||\alpha \mathbf{x}|| = |\alpha| ||\mathbf{x}||$ for any scalar $\alpha$.
3. **Triangle inequality**: $||\mathbf{x} + \mathbf{y}|| \leq ||\mathbf{x}|| + ||\mathbf{y}||$ for any vectors $\mathbf{x}$ and $\mathbf{y}$.

$\mathcal{l}_p \text{ norm: }  ||\mathbf{x}||_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$

**Matrix Norms** :
The Frobenius norm of a matrix $\mathbf{X}$ is defined as:

$$||\mathbf{X}||_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |x_{ij}|^2} = \sqrt{\text{trace}(\mathbf{X}^T \mathbf{X})}$$


## Projections and Orthogonality
<!--
Orthogonal vectors and projections onto subspaces
Gram-Schmidt process -->