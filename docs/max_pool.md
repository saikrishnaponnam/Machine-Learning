# Max Pooling
[Code](https://github.com/saikrishnaponnam/Machine-Learning/blob/main/src/layers/max_pool.py)

Max pooling is a down-sampling operation commonly used in convolutional neural networks (CNNs). It reduces the spatial dimensions of the input feature map while retaining the most important information. The operation works by sliding a window (or kernel) over the input feature map and taking the maximum value within that window.

## Forward Pass

During the forward pass, max pooling takes the maximum value from each window of the input feature map. The window size and stride determine how much the window moves across the input feature map.

Pooling window size: $k \times k$  
Stride: $s$

$$ Y_{i,j} = \max_{(m,n) \in R_{i,j} } X_{m,n} $$

where $R_{i,j}$ is the receptive field of size $k \times k$ starting at $(i.s,~j.s)$.

## Backward Pass

During the backward pass, max pooling propagates the gradient only to the positions that were selected during the forward pass. The gradient is set to zero for all other positions.

Let 
- $\nabla Y$ be the gradient from next layer
- $\nabla X$ to be backpropagated to previous(input) layer

$$\nabla X_{m,n} = \begin{cases} 
\nabla Y_{i,j}, \quad \text{if } X_{m,n} = Y_{i,j} ~\text{and}~ (m,n) \in R_{i,j} \\ 
0, \quad \text{ otherwise} 
\end{cases}$$

If multiple elements in a region are equal to the max, gradients are usually assigned arbitrarily or split.


