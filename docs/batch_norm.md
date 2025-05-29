# Batch Norm
[Paper](https://arxiv.org/abs/1502.03167) | [Code](https://github.com/saikrishnaponnam/Machine-Learning/blob/main/src/layers/batch_norm.py)

A technique used in deep learning to improve training speed, stability, and performance by normalizing the inputs to each layer. It helps mitigate issues like vanishing/exploding gradients and allows for higher learning rates.

Training a deep neural network is challenging due to issues such as:

- Internal covariate shift: The distribution of inputs to each layer changes during training as weights are updated.
- Vanishing/exploding gradients: Gradients can become too small or too large, making training difficult.
- Training instability: Due to these shifts and gradient issues, network may train slowly or diverge.

BatchNorm addresses these by normalizing the input to each layer, so the distribution remains more stable during training.

## Forward pass
For each mini-batch, BatchNorm normalizes the actications.  
Let input to a layer be $x \in \mathbb{R}^{m \times d}$ where m is the batch size and d is the number of features.  
For each feature $x_j$, compute:

1. Compute the mean: $\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_{ij}$
2. Compute the variance: $\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_{ij} - \mu_j)^2$
3. Normalize the feature: $\hat{x}_{j} = \frac{x_{j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$  
   where $\epsilon$ is a small constant to avoid division by zero.
4. Scale and shift: $y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j \\
    y_{j} = \gamma \hat{x}_j + \beta$  
   where $\gamma_j$ and $\beta_j$ are learnable parameters for each feature.

## Backward pass

During backpropagation, gradients are computed for $\gamma$ and $\beta$ as well.  


$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^m \frac{\partial L}{\partial y_i}\frac{\partial y_i}{\partial \gamma} = \sum_{i=1}^m \delta_i \hat{x}_i  \\  
\frac{\partial L}{\partial \beta} = \sum_{i=1}^m \frac{\partial L}{\partial y_i}\frac{\partial y_i}{\partial \beta} = \sum_{i=1}^m \delta_i$$

$$ \begin{align*} \frac{\partial L}{\partial x_i} &= \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial x_i} \\
\frac{\partial L}{\partial \hat{x}_j} &= \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial \hat{x}_j} = \delta_j \gamma \\
\frac{\partial \hat{x}_j}{\partial x_i} &= \frac{1}{\sigma} \frac{\partial (x_j - \mu)}{\partial x_i} + (x_j - \mu) \frac{\partial 1 / \sigma }{\partial x_i} \quad \text{Let }~ \sigma = \sqrt{\sigma^2 + \epsilon}  \\
\frac{\partial \hat{x}_j}{\partial x_i} &= \frac{1}{\sigma} \frac{\partial (x_j - \mu)}{\partial x_i} - \frac{x_j - \mu}{2 \sigma^3 } \frac{\partial \sigma^2 }{\partial x_i} \\
\frac{\partial \sigma^2}{\partial x_i} &= \sum_{j=1}^m \frac{\partial (x_j^2 + \mu^2 - 2x_j\mu )}{\partial x_i} =  \frac{2}{N} (x_i - \mu) ) \\
\frac{\partial \hat{x}_j}{\partial x_i} &= \frac{1}{\sigma} \frac{\partial (x_j - \mu)}{\partial x_i} - \frac{(x_j - \mu)(x_i - \mu)}{N \sigma^3 } 
\end{align*}$$

For j = i

$$ \frac{\partial \hat{x}_i}{\partial x_i} = \frac{m-1}{m \sigma} - \frac{(x_i - \mu)^2}{N \sigma^3 } \\
\frac{\partial L}{\partial x_i}= \delta_i \gamma (\frac{m-1}{m \sigma} - \frac{(x_i - \mu)^2}{N \sigma^3 })  $$

For j != i

$$\frac{\partial \hat{x}_i}{\partial x_i} = \frac{-1}{m\sigma} - \frac{(x_j - \mu)(x_i - \mu)}{N \sigma^3 } \\
\frac{\partial L}{\partial x_i}= \sum_{i \ne j} \delta_j \gamma (\frac{-1}{m\sigma} - \frac{(x_j - \mu)(x_i - \mu)}{N \sigma^3 }) $$

Combining both cases, we get:

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m \sigma}(m\delta_i - \sum_j \delta_j - \hat{x}_i \sum_j \delta_j \hat{x}_j)$$


### Advantages

- Keeps input distributions stable during training.
- Allows higher learning rates.
- Acts as a regularizer (due to added noise from mini-batch stats).

### Practical Considerations

- During inference, we use running averages of the mean and variance computed during training instead of batch statistics.
- Use after fully connected or convolutional layers, before activation functions.
- Initialization of $\gamma$ and $\beta$ is typically set to 1 and 0 respectively.

#### When not to use

- In small batch sizes, where batch statistics may not be reliable.
- In RNNs or LSTMs, where temporal dependencies are crucial.

## Q&A
1. How does BatchNorm affect the internal covariate shift?
    - Internal covariate shift refers to the change in the distribution of inputs to a layer during training as parameters of previous layers change. BatchNorm addresses this by normalizing the input of each layer, reducing the variability and stabilizing the input distribution. This leads to faster and more stable training.

2. How does BatchNorm affect gradient flow during backpropagation?
    - By normalizing inputs, it maintains well-behaved gradients, preventing exploding or vanishing gradients.

3. How does BatchNorm act as a form of regularization?
   - During training, BatchNorm uses mini-batch statistics (mean and variance), which introduces stochasticity/noise. This noise has a regularizing effect, somewhat similar to dropout, because it makes the model less likely to overfit to training data.

4. Why does BatchNorm often reduce the need for dropout?
    - Because BatchNorm already introduces noise through mini-batch statistics, it provides a regularizing effect. In practice, adding dropout on top of BatchNorm sometimes hurts performance or is redundant
   
5. What happens if you remove BatchNorm from a pretrained model?
    - Removing BatchNorm disrupts the learned activations and distributions, leading to poor performance. The weights were trained with normalized inputs in mind, so without BatchNorm, the distribution shift can cause instability and degraded accuracy unless you fine-tune the model extensively.

6. Can BatchNorm harm performance in some scenarios? When and why?
    - Small batch sizes, Online/incremental training, RNNs

7. What would happen if you froze the BatchNorm statistics during training?
    - Freezing (i.e., not updating running mean/variance) turns BatchNorm into a form of fixed normalization. This can:
        - Stabilize training in transfer learning or fine-tuning. 
        - Hurt performance if the frozen stats are not representative of the new data distribution, leading to mismatch between expected and actual input distributions at inference.

8. How does BatchNorm relate to whitening?
   - Whitening removes correlations between features. BatchNorm only normalizes each feature independently

