# Calculus

## Derivatives
A derivative is the rate of change in a function with respect to changes in its arguments. 
Derivatives can tell us how rapidly a loss function would increase or decrease were we to increase or decrease each parameter by an infinitesimally small amount

$$ f'(x) = \lim_{h \rightarrow 0 } \frac{f(x + h) - f(x)}{h} $$

## Chain rule
Used to compute the derivative of a composite function, i.e., a function that is formed by combining two or more functions.

$$y=f(u), \text{and }~ u = g(x)\\
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(u) \cdot g'(x)$$


## Partial Derivatives
The partial derivative of a function with respect to one of its arguments is the derivative of that function with respect to that argument, holding all other arguments constant.

Suppose that the input of function $f:\mathbb{R}^n \rightarrow \mathbb{R}$ is an n-dimensional vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$.
The gradient of the function $f$ with respect to $\mathbf{x}$ is a vector of $n$ partial derivatives:

$$\nabla_n f(\mathbf{x}) = [\partial_{x_1}f(\mathbf{x}), \partial_{x_2}f(\mathbf{x}), \dots, \partial_{x_n}f(\mathbf{x})]^T $$


### Total derivative
The **total derivative** of a function accounts for how the function changes with respect to all of its input variables, especially when each input depends on another variable (typically time `t`).

Suppose you have a function:$z = f(x_1, x_2, \dots, x_n)$
where each $ x_i = x_i(t) $, i.e., each variable depends on a common variable $t$.

The **total derivative** of $f$ with respect to $t$ is:

$$\frac{df}{dt} = \frac{\partial f}{\partial x_1} \frac{dx_1}{dt} + \frac{\partial f}{\partial x_2} \frac{dx_2}{dt} + \cdots + \frac{\partial f}{\partial x_n} \frac{dx_n}{dt}$$




## Matrix derivatives

 - $\nabla_{\mathbf{x}} A\mathbf{x} = A^T $
 - $\nabla_{\mathbf{x}} \mathbf{x}^TA\mathbf{x} = (A + A^T)\mathbf{x} $
 - $\nabla_{\mathbf{X}}||X||_F^2 = 2X$



## Hessian

Assume that the input of a function is a k-dimensional vector and its output is a scalar, so its Hessian matrix will have k eigenvalues. The solution of the function could be a local minimum, a local maximum, or a saddle point at a position where the function gradient is zero:

- When the eigenvalues of the function’s Hessian matrix at the zero-gradient position are all positive, we have a local minimum for the function.
- When the eigenvalues of the function’s Hessian matrix at the zero-gradient position are all negative, we have a local maximum for the function.
- When the eigenvalues of the function’s Hessian matrix at the zero-gradient position are negative and positive, we have a saddle point for the function.

[//]: # (TODO:)
https://d2l.ai/chapter_optimization/convexity.html