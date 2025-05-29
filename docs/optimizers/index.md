# Optimizers

## Optimization 
In any ML/DL model, we will define a loss function. We use optimization algorithms in attempt to minimize the loss.
In optimization, a loss function is often referred to as the objective function of the optimization problem.
The goal of optimization is to reduce the training error. However, the goal of DL is to reduce generalization error.
To accomplish the latter we need to pay attention to overfitting in addition to using the optimization algorithm to reduce the training error.

### Challenges in DL
In deep learning, most objective functions are complicated and do not have analytical solutions. Instead, we must use numerical optimization algorithms.
Some of the most vexing challenges  are local minima, saddle points, and vanishing gradients.

**Local minima:** If the value of $f(x)$ at $x$ is smaller than values at any other points in vicinity of $x$, then $f(x)$ could be local minima.
If $f(x)$ is minimum in entire domain it is called global minimum. 

When the numerical solution of an optimization problem is near the local optimum, the numerical solution obtained by the final iteration may only minimize the objective function locally, rather than globally, as the gradient of the objective function’s solutions approaches or becomes zero.

**Saddle Point:** A saddle point is any location where all gradients of a function vanish but which is neither a global nor a local minimum. 
Optimization might stall at this point, even though it is not a minimum. For high-dimensional problems the likelihood that at least some of the eigenvalues are negative is quite high. This makes saddle points more likely than local minima.



## RMSProp