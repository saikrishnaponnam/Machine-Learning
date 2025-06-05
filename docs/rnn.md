# RNN

For models like logistic regression and multilayer perceptron (MLP), the input $x_i$ is assumed to be a fixed size vector in $\mathbb{R}^d$. 
Such datasets are referred to as tabular datasets.

Image data, on the other hand, consists of a grid of pixels and can be represented as a 2D matrix. 
CNNs are used to process the spatial structure of images. 
However, these data types are still of fixed length.

But how do we handle data that is not of fixed length, such as a sequence of images in a video or a sequence of words in language tasks like image captioning or translation?

Recurrent Neural Networks (RNNs) are designed to process sequential data, where the input length can vary. 
RNNs capture the dynamics of sequences through recurrent connections, allowing them to maintain a memory of previous inputs.

## Working with sequences

A sequence consists of an ordered list of feature vectors $x_1, \dots, x_T$, where each $x_t \in \mathbb{R}^d$ is indexed by time step $t \in \mathbb{Z}^+$.

While individual inputs are typically assumed to be independently sampled from a distribution $P(X)$, in sequential data, we cannot assume that each time step is independent of the previous ones.

Given a sequential input, the goal may be to predict a single output $y$ or sequentially structured output ($y_1, \dots, y_T$).

## RNNs
An RNN processes sequential input one step at a time, maintaining a hidden state that serves as memory.
Let's denote:

- $x_t \in \mathbb{R}^d$: as the input at time step $t$
- $h_t \in \mathbb{R}^d$: as the hidden state at time step $t$
- $y_t \in \mathbb{R}^m$: as the output at time step $t$
- $W_{xh}$: as the weight matrix for input to hidden state
- $W_{hh}$: as the weight matrix for hidden state to hidden state
- $W_{hy}$: as the weight matrix for hidden state to output
- $\phi$: as the activation function

The hidden state $h_t$ at time step $t$ is computed from the previous hidden state $h_{t-1}$ and the current input $x_t$.
This hidden state stores information about the sequence up to that point. 
Unlike traditional deep neural networks, RNNs share the same parameters across all time steps.


<figure markdown="span">
    ![img.png](images/rnn.png)
    <figcaption>RNN unrolled</figcaption>
</figure>

$$h_t = \phi(a_t) = \phi(W_{xh}x_t + W_{hh} h_{t-1} + b_h ) \\
y_t = W_{hy} h_t + b_y$$

![rnn_types.png](images/rnn_types.png)

Not all RNNs produce output at every time step; some only output at the final time step. 
The key feature of an RNN is its hidden state, which captures information about the sequence.

## Training RNNs

Training RNNs is similar to training other neural networks, but with some additional considerations due to the sequential nature of the data.
### Loss Function
The loss function for RNNs can be defined as the sum of the losses at each time step, or as the average loss across all time steps.

$$L = \frac{1}{T} \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)$$

where $L$ is the loss function (e.g., cross-entropy loss), $y_t$ is the true output at time step $t$, and $\hat{y}_t$ is the predicted output at time step $t$.

## Backpropagation through time (BPTT)
When training RNNs, we often use backpropagation through time (BPTT) to compute gradients.
BPTT is an extension of backpropagation that allows us to compute gradients for sequences by unrolling the RNN through time, one step at a time.
The unrolled RNN is treated as a feedforward network, where each time step corresponds to a layer in the network.
But same parameters are shared across all time steps/layers. The gradients are computed for each time step and then accumulated to update the parameters.

Let $\delta_y^t = \frac{\partial L_t}{\partial \hat{y}_t}$, $\delta_h^t = \frac{\partial L_t}{\partial h_t}$, $\delta_a^t = \delta_h^t * \phi'(a_t) $, where $\phi'(a_t)$ is the derivative of the activation function element-wise.

### Gradients w.r.t output weights

$$\begin{align*}
\frac{\partial L}{\partial  W_{hy}} &= \sum_t \frac{\partial L_t}{\partial W_{hy}} \\
\frac{\partial L_t}{\partial W_{hy}} &= \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial w_{hy}} = \delta_y^t . (h_t)^T \\
\frac{\partial L}{\partial W_{hy}} &= \sum_t {\delta_y^t} \cdot (h_t)^T \\
\end{align*}$$

### Gradients w.r.t input weights
To update the input-to-hidden weights $W_{xh}$, we compute the gradient of the loss with respect to $W_{xh}$ by accumulating contributions from all time steps:

$$\begin{align*}
\frac{\partial L}{\partial  W_{xh}} &= \sum_t \frac{\partial L}{\partial  a_t} \cdot \frac{\partial a_t}{\partial  W_{xh}} \\
\frac{\partial a_t}{\partial  W_{xh}} &= (x_t)^T\\
\frac{\partial L}{\partial  a_t} &= \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial a_t} = \frac{\partial L}{\partial h_t} * \phi'(a_t) \\ 
\frac{\partial L}{\partial  h_t} &= \frac{\partial L_t}{\partial h_t} + \frac{\partial L}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t} \\
\frac{\partial L}{\partial  h_t} &= \frac{\partial L_t}{\partial h_t} + \frac{\partial L}{\partial h_{t+1}} * \phi(a_{t+1}) \cdot W_{hh}^T \\
\frac{\partial L}{\partial w_{xh}} &= \sum_t \delta_h^t * \phi'(a_t) \cdot (x_t)^T \\ 
\text{where } \delta_h^t &= \frac{\partial L_t}{\partial h_t} + \delta_h^{t+1} * \phi'(a_{t+1}) \cdot (w_{hh})^T = \frac{\partial L_t}{\partial h_t} + \frac{\partial L}{\partial a_{t+1}} \cdot (w_{hh})^T   \\
\end{align*}$$

We get $\frac{\partial L_t}{\partial h_t}$ from the output ($y_t$ - Linear layer) or sometimes directly from loss function.

### Gradients w.r.t recurrent weights

Similar to above

$$\begin{align*}
\frac{\partial L}{\partial  W_{hh}} &= \sum_t \frac{\partial L}{\partial  a_t} \cdot \frac{\partial a_t}{\partial  W_{hh}} \\
\frac{\partial a_t}{\partial  W_{hh}} &= (h_{t-1})^T\\
\frac{\partial L}{\partial w_{hh}} &= \sum_t \delta_h^t * \phi'(a_t) \cdot (h_{t-1})^T \\
\end{align*}$$

### Gradients w.r.t input

$$\begin{align*}
\frac{\partial L}{\partial  x_t} &= \sum_{t} \frac{\partial L}{\partial  a_t} \cdot \frac{\partial a_t}{\partial x_t} \\
\frac{\partial L}{\partial  x_t} &= \sum_{t} \frac{\partial L}{\partial  a_t} \cdot (W_{xh})^T \\
\end{align*}$$

### Vanishing and Exploding Gradients
RNNs are susceptible to vanishing and exploding gradients, which can make training difficult for long sequences.

## Gradient Clipping
Gradient clipping is a technique used to mitigate exploding gradients by capping the gradients during backpropagation.

## Advantages 

## Disadvantages