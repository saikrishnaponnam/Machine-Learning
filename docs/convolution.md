# Convolution

[Paper 1](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 
| [Code](https://github.com/saikrishnaponnam/Machine-Learning/blob/main/src/layers/conv.py)

Input: $\mathbf{X} \in \mathbb{R}^{C_{in} \times H_{in} \times W_{in} }$  
Kernel: $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_H \times K_W}$  
Output: $\mathbf{Y} \in \mathbb{R}^{C_{out} \times H_{out} \times W_{out}}$  
$\delta = \frac{\partial L}{\partial Y} \in \mathbb{R}^{C_{out} \times H_{out} \times W_{out}}$

$H_{out} = \frac{H_{in} + 2P - K}{stride} + 1,~ W_{out} = \frac{W_{in} + 2P - K}{stride} + 1$

## Forward pass

stride = 1, padding = 0

$$Y_{c,i,j} = \sum_{k=0}^{C_{in}-1} \sum_{m=0}^{K_H-1} \sum_{n=0}^{K_W-1} W_{c,k,m,n}.X_{k, i+m, j+n} + b_c$$

Vectorized form:

$$\tilde{W} \in \mathbb{R}^{C_{out} \times (C_{in}.K_H.K_W)} \\
\text{Unfold Input: }\tilde{X} \in \mathbb{R}^{(C_{in}.K_H.K_W) \times (H_{out}.W_{out})} \\
Y = \tilde{W} \cdot \tilde{X} + b \\
$$

## Backward pass
**Gradients w.r.t Weights:**

$$\frac{\partial L}{\partial W_{c,k,m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{c,i,j}} \frac{\partial Y_{c,i,j}}{\partial W_{c,k,m,n}} \\
\frac{\partial Y_{c,i,j}}{\partial W_{c,k,m,n}} = X_{k,i+m,j+n} \\
\frac{\partial L}{\partial W_{c,k,m,n}} = \sum_{i,j} \delta_{c,i,j} X_{k,i+m,j+n}$$

$$ \frac{\partial L}{\partial \tilde{W}} = \delta \tilde{X}^T $$

**Gradients w.r.t bias:**

$$
\frac{\partial L}{\partial b_c} = \sum_{i,j} \frac{\partial L}{\partial Y_{c,i,j}} \frac{\partial Y_{c,i,j}}{\partial b_c} \\
\frac{\partial Y_{c,i,j}}{\partial b_c} = 1 \\
\frac{\partial L}{\partial b_c} = \sum_{i,j} \delta_{c,i,j}
$$

**Gradients w.r.t input:**

$$\frac{\partial L}{\partial X_{k,p,q}} = \sum_{c=0}^{C_{out} - 1} \sum_{i=0}^{H_{out} - 1} \sum_{j=0}^{W_{out} - 1} \frac{\partial L}{\partial Y_{c,i,j}} \frac{\partial Y_{c,i,j}}{\partial X_{k,p,q}} $$
$Y_{c,i,j}$ depends on $X_{k,p,q}$ iff: $p=i+m,~ q=j+n$ 

$$\frac{\partial Y_{c,i,j}}{\partial X_{k,p,q}} = \begin{cases}
W_{c,k,p-i,q-j} & \text{if } 0 \le p-i \lt K_H \text{ and } 0 \le q-j \lt K_W \\ 0 & \text{otherwise}
\end{cases} \\
\frac{\partial L}{\partial X_{k,p,q}} = \sum_{c=0}^{C_{out} - 1} \sum_{i=0}^{H_{out} - 1} \sum_{j=0}^{W_{out} - 1} \delta_{c,i,j} W_{c,k,p-i,q-j}$$

This is equivalent to a full convolution of the gradient $\delta$ with the flipped kernel.
\\$flip(W_{c,k,m,n}) = W_{c,k,K_H-1-m, K_W-1-n}$

$$ \frac{\partial L}{\partial \tilde{X}} = \tilde{W}^T . \delta $$
