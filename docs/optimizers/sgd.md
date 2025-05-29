# SGD
[Paper 1](https://arxiv.org/pdf/1609.04747) | [Paper 2](https://proceedings.mlr.press/v28/sutskever13.html) | [Code](https://github.com/saikrishnaponnam/Machine-Learning/blob/main/src/optimizer/adam.py)

The objective function is average is  of the loss over the training dataset. The gradient of the objective function is also calculated as average of gradients at each training sample.
Parameters are updated using this gradient. This is called Gradient descent.
The computational cost of GD is O(n), which grows linearly with n. Therefore, when training dataset is larger, cost of gradient descent for each iteration is higher.
This also delays convergence. Because GD is deterministic, it may get stuck in bad local minima or saddle points.

In Stochastic gradient descent(SGD) we use a single datapoint sampled from training dataset to compute gradient and update parameters. 
the stochastic gradient $\nabla J_i(w)$ is an unbiased estimate of the full gradient $\nabla J(w)$ because 

$$\mathbb{E}_i \nabla J_i(w) = \frac{1}{n} \sum_i \nabla J_i(w) = \nabla J(w)$$

This means that, on average, the stochastic gradient is a good estimate of the gradient.
The randomness introduces a kind of jitter that can help the model escape shallow minima or saddle points.

## Minibatch SGD
SGD is computationally inefficient as it doesn't fully utilize the CPU or GPU. We can increase the computational efficiency of this operation by applying it to a minibatch of observations at each iteration.
SGD has high variance is updates making it noisy and less stable convergence. May not converge smoothly or at all unless you carefully schedule learning rate.

In Mini-batch SGD instead of one datapoint we sample a small batch (32, 64 or 128) and average the gradients over the batch.
This reduces variance compared to plain SGD making updates more stable. 


## Momentum

When loss changes quickly in one direction and slowly in another direction. 
There will be very slow progress along the shallow dimension and jitter along the steep direction.

To avoid these conditions we add a momentum to the SGD to build up velocity or friction.

$$ v_{t+1} = \rho v_t + \nabla J(w_t) \\
w_{t+1} = w_t - \alpha v_{t+1}
$$

where $\rho$ is typically 0.9 or 0.99

Momentum accumulates past gradients to continue moving in the same direction, smoothing the path toward the minimum and accelerating convergence.

**Nesterov Momentum**
A variant of momentum where gradients are computed at lookahead point, which we get using velocity.
$$ v_{t+1} = \rho v_t + \nabla J(w_t - \rho v_t)  $$


## Q&A
1. Can momentum cause the optimization to overshoot the minimum? Why or why not?
   - Yes, if $\rho$ is too high and lr is also high, velocity can cause the update move away from minimum.
2. Why does momentum help reduce oscillations during training?
    - In areas with steep but narrow valleys gradients can change directions quickly. Momentum averages recent gradients, dampening the effect of sudden direction changes.
3. If your training loss is oscillating wildly, how might adjusting the momentum coefficient help?
    - Increase momentum coefficient slightly - it will smooth out updates. Alternatively, lower the lr.
4. How would you tune momentum and learning rate together during hyperparameter optimization?
    - Use random search across $\alpha \in {0.1, 0.01, 0.001, 0.0001}$ & $\rho \in {0.8, 0.9, 0.95, 0.99}$
    - Tools like Optuna, Ray tune are useful in automating this.
5. Momentum helps in faster convergence, but can it affect generalization? How?
    - Yes, momentum can affect generalization, both positively and negatively.
      - By smoothing the trajectory, momentum helps avoid overfitting to local noise in the data.
      - If momentum + lr causes too aggressive convergence, it might settle in sharp minima — which tend to overfit.
