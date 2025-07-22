# Dot Product Attention

Paper: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

## Introduction

Neural Machine Translation (NMT) has gained popularity due to its minimal reliance on domain-specific knowledge and its ability to avoid the need for large phrase tables, which are common in SMT models. The introduction of attention mechanisms has further improved NMT by enabling models to learn alignments between different modalities, such as images and text or speech and text. Notably, [Bahdanau et al.](additive_attention.md) demonstrated that attention allows NMT models to focus on the most relevant parts of the source sentence during translation, leading to better performance.

This paper investigates two new types of attention mechanisms for NMT: global and local attention. It also explores various alignment functions for computing attention scores, including dot product, general, and concat functions.

## Attention Mechanism

Attention mechanisms can be classified into two main types: global and local attention. Both approaches use an encoder-decoder architecture, but they differ in how the context vector is computed in the decoder.

The encoder processes the input sentence and produces a set of hidden states, denoted as $s$. The decoder then generates each target word step by step, using the previous hidden state $h_{j-1}$ and $s$. 
After getting the context vector, both attention types follow the same steps. 

- Compute attentional hidden state from hidden state $h_t$ and context vector $c_t$:  
    $\tilde{h}_t = tanh(W_h [h_t;c_t])$
- Pass the attentional vector through a softmax layer to produce the output distribution over the target vocabulary: $p(y_t | y_{<t}, x) = softmax(W_s \tilde{h}_t)$


### Global Attention

Global attention computes the context vector by considering all positions in the source sequence. For each target word, the model calculates an alignment vector a variable-length alignment vector $a_t$, whose size equals input sentence length, by comparing current target hidden state $h_t$ and input hidden states $\bar{h}_j$:

$$a_{tj} = align(h_t, \bar{h}_j) = \frac{exp(score(h_t, \bar{h}_j))}{\sum_{k=1}^{T} exp(score(h_t, \bar{h}_k))}$$

Score is referred as content-based function, for which the paper proposes three variants: dot product, general, and concat.

$$ score(h_t, \bar{h}_j) = 
\begin{cases}
h_t^T \bar{h}_j & \text{(dot)} \\
h_t^T W_a \bar{h}_j & \text{(general)} \\
v_a^T tanh(W_a [h_t; \bar{h}_j]) & \text{(concat)}
\end{cases} $$


Given the alignment vector $a_t$, the context vector $c_t$ is computed as a weighted sum of the source hidden states.

This approach is similar to [Additive Attention](additive_attention.md), but differs in that it uses only the last layer's hidden state in a stacked LSTM, whereas the additive approach uses a non-stacked BiLSTM and a concatenation-based alignment function.

### Local Attention

While global attention is effective, it can be computationally expensive for long sequences. Local attention addresses this by focusing only on a small window of source positions for each target word.

Local attention selectively focuses on a small window of context. For each target word, the model predicts an aligned position $p_t$ in the source sequence. The context vector is then computed using only the source hidden states within a window centered at $p_t$ i.e., $[p_t - D, p_t + D]$, where $D$ is empirically selected. The alignment position $p_t$ can be derived in 2 ways:

- **Monotonic alignment (local-m):** $p_t = t$, assuming input and target are roughly monotonically aligned.
- **Predictive alignment (local-p):** $p_t = S. sigmoid(v_p^T tanh(W_p h_t))$, where $S$ is the length of the source sentence.

To favour alignments near $p_t$ we use a Gaussian distribution to compute the alignment weights:

$$a_{tj} = align(h_t, \bar{h}_j) exp(-\frac{(j - p_t)^2}{2\sigma^2})$$

where $\sigma$ is empirically set to $D/2$.

### Input Feeding approach

In standard MT, a coverage set is maintained to keep track of translated input words. But attentions decisions are made independently which is suboptimal. The attention decisions should be made jointly taking into account the past alignment information. To address this, the attentional vectors $\tilde{h}_t$ are concatenated with inputs at the next steps. This has two effects:

- Hope model will be fully aware of the past alignment choices.
- A very deep network spanning horizontally and vertically is created

## Training

The models are trained on the WMT'14 dataset, which contains 4.5 million sentence pairs. The vocabulary is limited to the 50,000 most frequent words in each language.

- The stacked LSTM model consists of 4 layers, each with 1,000 hidden units and 1,000-dimensional embeddings.
- Model parameters are initialized uniformly in the range $[-0.1, 0.1]$.
- Training is performed for 10 epochs using SGD with an initial learning rate of 0.1. After 5 epochs, the learning rate is halved after each subsequent epoch.
- Dropout with a probability of 0.2 is applied, and in this case, the model is trained for 12 epochs.

## Experiments

Experiments are conducted on the WMT English-German translation task in both translation directions. The paper compares various attention mechanisms (global, local-m, local-p) and alignment functions (dot, general, concat). Results demonstrate that attention-based models consistently outperform standard encoder-decoder models, particularly on longer sentences.

## Results

Attention-based models achieve substantial improvements in BLEU scores compared to baseline NMT models. Among the tested approaches, local attention with predictive alignment (local-p) often yields the best performance. Additionally, these models produce more interpretable alignments, as shown by the attention weight visualizations (figure 7).

### Choice of Alignment Function

The results indicate that the dot product alignment works well for global attention, while the general alignment function performs better for local attention.
