# Adam

[Paper](https://arxiv.org/abs/1412.6980) | [Code](https://github.com/saikrishnaponnam/Machine-Learning/blob/main/src/optimizer/adam.py)

In short for Adaptive Moment Estimation. It combines the advantages of two other widely used extensions SGD: AdaGrad and RMSProp.

Adam uses adaptive learning rates for each parameter, which helps in dealing with sparse gradients and noisy data. 
It also incorporates momentum to smooth out updates.

## How Adam works
Adam maintains two moving averages for each parameter:

1. **First moment (mean)**: This is the exponentially weighted average of past gradients.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

2. **Second moment (variance)**: This is the exponentially weighted average of past squared gradients.

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

where $g_t = \nabla J(w_t), \quad \beta_1, \beta_2 \in [0,1)$

3. **Bias correction**: Because these averages are biased towards zero, especially in the initial steps, we apply bias correction

$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

4. **Parameter update**:

$$ w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

## Intution

- $m_t$: Like momentum, it helps smooth out the updates by considering past gradients.
- $v_t$: Similar to RMSProp, it normalizes the updates by the variance of past gradients, allowing for adaptive learning rates.


## Cons

- Sometimes generalizes worse than SGD.
- May converge to sharp minima, which can lead to overfitting.

## When to use Adam vs SGD
- **Adam**: 
  - Working with sparse data, such as NLP tasks (transformers, LSTMs).
  - Fast convergence or less hyperparameter tuning.
  - Prototype quickly
  - Imbalanced or noisy datasets.
- **SGD**:
  - When you have a lot of data and can afford longer training times.
  - Prioritize generalization over speed.
  - Production-level training.
  - Need minimal memory overhead.

## Q&A